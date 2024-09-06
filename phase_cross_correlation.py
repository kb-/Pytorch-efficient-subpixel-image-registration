import itertools
import warnings
from math import ceil
from typing import Optional, Union

import torch
from numpy import fix
from torch.fft import fftfreq, fftn, ifftn


def _compute_phasediff(cross_correlation_max: torch.Tensor) -> torch.Tensor:
    """
    Compute global phase difference between the two images (should be
    zero if images are non-negative).

    Parameters
    ----------
    cross_correlation_max : torch.Tensor (complex)
        The complex value of the cross-correlation at its maximum point.

    Returns
    -------
    phasediff : torch.Tensor
        The global phase difference between the two images.
    """
    # Use torch.atan2 to compute the phase difference
    return torch.atan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(
    cross_correlation_max: torch.Tensor, src_amp: float, target_amp: float
) -> torch.Tensor:
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.

    Parameters
    ----------
    cross_correlation_max : torch.Tensor (complex)
        The complex value of the cross-correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image.
    target_amp : float
        The normalized average image intensity of the target image.

    Returns
    -------
    error : torch.Tensor
        The RMS error between the two images.
    """
    amp = src_amp * target_amp
    if amp == 0:
        warnings.warn(
            "Could not determine RMS error between images with the normalized "
            f"average intensities {src_amp!r} and {target_amp!r}. Either the "
            "reference or moving image may be empty.",
            UserWarning,
            stacklevel=3,
        )
        # Return NaN if amplitudes are zero
        return torch.tensor(float("nan"), device=cross_correlation_max.device)

    # Compute the error using torch operations
    error = 1.0 - (cross_correlation_max * torch.conj(cross_correlation_max)) / amp
    return torch.sqrt(torch.abs(error))


def _upsampled_dft(
    data: torch.Tensor,
    upsampled_region_size: int | tuple,
    upsample_factor: int = 1,
    axis_offsets: tuple[int] | None = None,
) -> torch.Tensor:
    """
    Perform an upsampled DFT by matrix multiplication.

    Parameters
    ----------
    data : torch.Tensor
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : int or tuple of ints
        The size of the region to be sampled. If one integer is provided,
        it is duplicated across all dimensions.
    upsample_factor : int, optional
        The upsampling factor. Defaults to 1.
    axis_offsets : tuple of ints, optional
        The offsets of the region to be sampled. Defaults to None (uses image center).

    Returns
    -------
    output : torch.Tensor
        The upsampled DFT of the specified region.
    """
    device = data.device

    # Expand integer input for `upsampled_region_size` to a list if necessary
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "Shape of upsampled region sizes must match the input data's number "
                "of dimensions."
            )

    # Set axis offsets to the center of the image by default
    if axis_offsets is None:
        axis_offsets = [0] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "Number of axis offsets must match the input data's number of "
                "dimensions."
            )

    im2pi = 1j * 2 * torch.pi

    # Iterate over each dimension and apply the DFT kernel for the specified upsample
    # factor
    # Zip together data.shape, upsampled_region_size, and axis_offsets
    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    # Iterate over these properties in reverse order
    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        # Create the kernel for the current dimension
        kernel = (torch.arange(ups_size, device=device) - ax_offset).unsqueeze(
            1
        ) * fftfreq(n_items, d=upsample_factor, device=device)
        kernel = torch.exp(-im2pi * kernel)

        # Ensure the kernel's precision matches the data
        kernel = kernel.to(dtype=data.dtype)

        # Apply the kernel along the current dimension using tensordot
        data = torch.tensordot(kernel, data, dims=([1], [-1]))  # -1 instead of dim

    return data


def _disambiguate_shift(
    reference_image: torch.Tensor, moving_image: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """
    Determine the correct real-space shift based on periodic shifts.

    Parameters
    ----------
    reference_image : torch.Tensor
        The reference (non-moving) image.
    moving_image : torch.Tensor
        The moving image (shifted). Must have the same shape as the reference image.
    shift : torch.Tensor
        The shift to apply to each axis of the moving image, modulo image size.

    Returns
    -------
    real_shift : torch.Tensor
        The disambiguated shift in real space.
    """
    shape = reference_image.shape
    dtype = reference_image.dtype
    device = reference_image.device

    # Compute positive and negative shifts
    positive_shift = [(shift_i % s) for shift_i, s in zip(shift, shape)]
    negative_shift = [shift_i - s for shift_i, s in zip(positive_shift, shape)]

    # Determine if subpixel interpolation is needed (when shift contains decimals)
    # subpixel = any(shift_i % 1 != 0 for shift_i in shift)
    # interp_order = 3 if subpixel else 0  # Cubic interpolation (order=3) if subpixel

    # Apply the shift to the moving image using grid-wrap mode (similar to torch.roll)
    shifted = torch.roll(
        moving_image, shifts=[int(s) for s in shift], dims=list(range(len(shift)))
    )

    # Get the rounded indices for slicing
    indices = torch.round(torch.tensor(positive_shift, device=device)).int()
    splits_per_dim = [(slice(0, i.item()), slice(i.item(), None)) for i in indices]

    max_corr = -1.0
    max_slice = None

    # Iterate over possible shifts and find the best cross-correlation
    for test_slice in itertools.product(*splits_per_dim):
        # Reshape the slices for cross-correlation
        reference_tile = reference_image[test_slice].flatten()
        moving_tile = shifted[test_slice].flatten()

        if reference_tile.numel() > 2:
            # Compute correlation (avoid division by zero errors using
            # `torch.nan_to_num`)
            reference_mean = torch.mean(reference_tile)
            moving_mean = torch.mean(moving_tile)
            reference_std = torch.std(reference_tile)
            moving_std = torch.std(moving_tile)

            if reference_std > 0 and moving_std > 0:
                corr = torch.mean(
                    (reference_tile - reference_mean) * (moving_tile - moving_mean)
                ) / (reference_std * moving_std)
            else:
                # Correlation invalid if std is 0
                corr = torch.tensor(-1.0, dtype=dtype, device=device)
        else:
            # Invalid correlation for small tiles
            corr = torch.tensor(-1.0, dtype=dtype, device=device)

        if corr.item() > max_corr:
            max_corr = corr.item()
            max_slice = test_slice

    # If no valid slice was found, return the original shift with a warning
    if max_slice is None:
        warnings.warn(
            f"Could not determine real-space shift for periodic shift {shift} "
            f"as requested by `disambiguate=True` (disambiguation is degenerate).",
            stacklevel=3,
        )
        return shift

    # Compute the final disambiguated shift
    real_shift_acc = []
    for sl, pos_shift, neg_shift in zip(max_slice, positive_shift, negative_shift):
        real_shift_acc.append(pos_shift if sl.stop is None else neg_shift)

    return torch.tensor(real_shift_acc, dtype=dtype, device=device)


def _masked_phase_cross_correlation(
    reference_image: torch.Tensor,
    moving_image: torch.Tensor,
    reference_mask: torch.Tensor,
    moving_mask: torch.Tensor = None,
    overlap_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Masked image translation registration by normalized cross-correlation.

    Parameters
    ----------
    reference_image : torch.Tensor
        Reference image.
    moving_image : torch.Tensor
        Image to register. Must have the same dimensionality as ``reference_image`` but
        not necessarily the same size.
    reference_mask : torch.Tensor
        Boolean mask for ``reference_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels.
    moving_mask : torch.Tensor, optional
        Boolean mask for ``moving_image``. If None, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for translations
        corresponding with an overlap ratio
        lower than this threshold will be ignored. A lower ratio leads to smaller
        maximum translation.

    Returns
    -------
    shifts : torch.Tensor
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``.
    """

    device = reference_image.device

    # If moving_mask is None, use reference_mask as the moving_mask
    if moving_mask is None:
        if reference_image.shape != moving_image.shape:
            raise ValueError(
                "Input images have different shapes, moving_mask must "
                "be explicitly set."
            )
        moving_mask = reference_mask.clone().bool()

    # Validate that images and masks have matching shapes
    for im, mask in [(reference_image, reference_mask), (moving_image, moving_mask)]:
        if im.shape != mask.shape:
            raise ValueError("Image sizes must match their respective mask sizes.")

    # Mask the images by applying the masks
    reference_image = reference_image * reference_mask
    moving_image = moving_image * moving_mask

    # Compute FFT of the images
    ref_fft = torch.fftn(reference_image)
    mov_fft = torch.fftn(moving_image)

    # Compute cross-correlation in the Fourier domain
    xcorr = torch.ifftn(ref_fft * torch.conj(mov_fft)).real

    # Apply masking to the cross-correlation (with overlap ratio consideration)
    mask_product = torch.ifftn(
        torch.fftn(reference_mask) * torch.conj(torch.fftn(moving_mask))
    ).real

    xcorr /= torch.maximum(mask_product, torch.tensor(overlap_ratio, device=device))

    # Find the maximum location of the cross-correlation
    maxima = torch.nonzero(xcorr == torch.max(xcorr))
    center = torch.mean(maxima.float(), dim=0)

    shifts = center - torch.tensor(reference_image.shape, device=device) + 1

    # Adjust the shift to account for size mismatch between images
    size_mismatch = torch.tensor(moving_image.shape, device=device) - torch.tensor(
        reference_image.shape, device=device
    )

    return -shifts + (size_mismatch / 2)


def phase_cross_correlation(
    reference_image: torch.Tensor,
    moving_image: torch.Tensor,
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: torch.Tensor = None,
    moving_mask: torch.Tensor = None,
    overlap_ratio: float = 0.3,
    normalization: Optional[str] = "phase",
) -> tuple[torch.Tensor, Union[torch.Tensor, float], Union[torch.Tensor, float]]:
    """
    Efficient subpixel image translation registration by cross-correlation.

    This function computes the shift between two images using phase cross-correlation.
    It estimates the translation shift between the reference and moving images, and can
    refine the estimate through upsampling the Discrete Fourier Transform (DFT).

    Parameters
    ----------
    reference_image : torch.Tensor
        Reference image.
    moving_image : torch.Tensor
        Image to register. Must have the same dimensionality as ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor for more precise registration.
        The default is 1 (no upsampling).
    space : str, optional
        Defines how the input data is interpreted. "real" means the data will be FFT'd,
        while "fourier" means the input is already in Fourier space. Defaults to "real".
    disambiguate : bool, optional
        If True, disambiguate the shift returned, which may have ambiguities due to the
        periodic nature of the Fourier transform. Defaults to False.
    reference_mask : torch.Tensor, optional
        Boolean mask for ``reference_image`` indicating valid pixels. If provided,
        masked cross-correlation will be used.
    moving_mask : torch.Tensor, optional
        Boolean mask for ``moving_image``. If None, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images when masks are provided.
        Defaults to 0.3.
    normalization : str, optional
        Normalization method for cross-correlation. Can be "phase" or None.
        Defaults to "phase".

    Returns
    -------
    shift : torch.Tensor
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``.
    error : torch.Tensor
        The RMS error between ``reference_image`` and ``moving_image``.
    phasediff : torch.Tensor
        Global phase difference between the two images (should be zero if both images
        are non-negative).
    """

    # Infer the dtype from the input tensors
    dtype = reference_image.dtype
    device = reference_image.device

    # If masks are provided, handle masked cross-correlation
    if reference_mask is not None or moving_mask is not None:
        shift = _masked_phase_cross_correlation(
            reference_image, moving_image, reference_mask, moving_mask, overlap_ratio
        )
        return shift, torch.nan, torch.nan

    # Ensure images are the same shape
    if reference_image.shape != moving_image.shape:
        raise ValueError("Images must have the same shape")

    # If the input is in Fourier space, assume it's already FFT transformed
    if space.lower() == "fourier":
        src_freq = reference_image
        target_freq = moving_image
    # Otherwise, compute the FFT of the input images
    elif space.lower() == "real":
        src_freq = fftn(reference_image)
        target_freq = fftn(moving_image)
    else:
        raise ValueError('space argument must be "real" or "fourier"')

    # Compute cross-correlation using IFFT of the product of the FFTs
    shape = src_freq.shape
    image_product = src_freq * torch.conj(target_freq)

    # Normalize if requested
    if normalization == "phase":
        eps = torch.tensor(
            torch.finfo(image_product.real.dtype).eps,
            dtype=image_product.real.dtype,
            device=device,
        )
        image_product /= torch.maximum(torch.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError("Normalization must be either 'phase' or None")

    # Inverse FFT to get cross-correlation in spatial domain
    cross_correlation = ifftn(image_product)

    # Find the location of the peak in the cross-correlation
    maxima = torch.argmax(torch.abs(cross_correlation))
    maxima = torch.unravel_index(maxima, cross_correlation.shape)

    midpoint = torch.floor(torch.tensor(shape, dtype=dtype, device=device) / 2)
    shift = torch.tensor(maxima, dtype=dtype, device=device)

    # Adjust shift based on whether it exceeded half the image size
    shift[shift > midpoint] -= torch.tensor(shape, dtype=shift.dtype, device=device)[
        shift > midpoint
    ]

    # If no upsampling is requested, compute the shift and return
    if upsample_factor == 1:
        src_amp = (
            torch.sum(torch.real(src_freq * torch.conj(src_freq))) / src_freq.numel()
        )
        target_amp = (
            torch.sum(torch.real(target_freq * torch.conj(target_freq)))
            / target_freq.numel()
        )
        CCmax = cross_correlation[maxima]
    else:
        # For upsampled cross-correlation, refine the shift estimate
        shift = torch.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = ceil(upsample_factor * 1.5)
        dftshift = fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor
        cross_correlation = _upsampled_dft(
            torch.conj(image_product),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()

        maxima = torch.argmax(torch.abs(cross_correlation))
        maxima = torch.unravel_index(maxima, cross_correlation.shape)
        CCmax = cross_correlation[maxima]

        maxima = torch.tensor(maxima, dtype=dtype, device=device)
        maxima -= dftshift

        shift += maxima / upsample_factor
        src_amp = torch.sum(torch.real(src_freq * torch.conj(src_freq)))
        target_amp = torch.sum(torch.real(target_freq * torch.conj(target_freq)))

    # Handle edge cases for rows/columns with only one element
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shift[dim] = 0

    # Disambiguate shift if necessary
    if disambiguate:
        if space.lower() != "real":
            reference_image = ifftn(reference_image)
            moving_image = ifftn(moving_image)
        shift = _disambiguate_shift(reference_image, moving_image, shift)

    # Check for NaN values and provide appropriate error message
    if torch.isnan(CCmax) or torch.isnan(src_amp) or torch.isnan(target_amp):
        raise ValueError(
            "NaN values found, please remove NaNs from your input data "
            "or use the reference_mask/moving_mask arguments."
        )

    return shift, _compute_error(CCmax, src_amp, target_amp), _compute_phasediff(CCmax)
