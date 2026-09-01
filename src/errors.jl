"""
    VkFFTError(code::Int, name::String)

An error reported by the VkFFT C library.

# Fields
- `code::Int`: The raw `VkFFTResult` code, as returned by the wrapper
- `name::String`: The name of the code, from VkFFT's own `getVkFFTErrorString`
"""
struct VkFFTError <: Exception
    code::Int
    name::String
end

Base.showerror(io::IO, err::VkFFTError) = print(io, "Fatal VkFFT error: $(err.name) (code: $(err.code))")
