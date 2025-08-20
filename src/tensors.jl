using BFloat16s

@enum GGMLType::UInt32 begin
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30
end

const ggml_type_to_type = Dict(
    F32 => Float32,
    F16 => Float16,
    I8 => Int8,
    I16 => Int16,
    I32 => Int32,
    I64 => Int64,
    F64 => Float64,
    BF16 => BFloat16,
)

const type_to_ggml_type = Dict(reverse(kv) for kv in ggml_type_to_type)

struct TensorInfo
    dims::Dims
    type::GGMLType
    offset::UInt64
end

function read_tensor_info(io::IO)
    n_dims = Int(read(io, UInt32))
    dims = ntuple(i -> Int(read(io, UInt64)), n_dims)
    gtype = GGMLType(read(io, UInt32))
    offset = read(io, UInt64)
    return TensorInfo(dims, gtype, offset)
end

function read_tensor_info_dict(io::IO, count::Integer)
    tensor_info_dict = OrderedDict{String,TensorInfo}()
    for _ in 1:count
        name = read_string(io)
        tensor_info = read_tensor_info(io)
        tensor_info_dict[name] = tensor_info
    end
    return tensor_info_dict
end

function get_tensor(tensor_data::AbstractVector{UInt8}, tensor_info::TensorInfo)
    T = @something get(ggml_type_to_type, tensor_info.type, nothing) error("Unsupported tensor type for reading: $(tensor_info.type)")

    byte_range = (tensor_info.offset + 1):(tensor_info.offset + prod(tensor_info.dims) * sizeof(T))
    tensor = reshape(reinterpret(T, view(tensor_data, byte_range)), tensor_info.dims)

    if tensor isa AbstractVector
        return tensor
    elseif tensor isa AbstractMatrix
        return transpose(tensor)
    else
        return PermutedDimsArray(tensor, Tuple(ndims(tensor):-1:1))
    end
end

function get_tensors(tensor_data::AbstractVector{UInt8}, tensor_info_dict)
    tensors = OrderedDict{String,AbstractArray}()
    for (name, info) in tensor_info_dict
        tensors[name] = get_tensor(tensor_data, info)
    end
    return tensors
end

function get_tensor_info_dict(tensors::OrderedDict{String,AbstractArray})
    tensor_info_dict = OrderedDict{String,TensorInfo}()
    offset = zero(UInt64)
    for name in keys(tensors)
        tensor = tensors[name]
        Δoffset = prod(size(tensor)) * sizeof(eltype(tensor))
        tensor_info_dict[name] = TensorInfo(reverse(size(tensor)), type_to_ggml_type[eltype(tensor)], offset)
        offset += Δoffset
    end
    return tensor_info_dict
end

function write_tensor_info_dict(io::IO, tensor_info_dict::OrderedDict{String,TensorInfo})
    for (name, tensor_info) in tensor_info_dict
        write_string(io, name)
        write(io, UInt32(length(tensor_info.dims)))
        write(io, UInt64[tensor_info.dims...])
        write(io, UInt32(tensor_info.type))
        write(io, tensor_info.offset)
    end
end

_permutedims(x::AbstractArray, perm) = PermutedDimsArray(x, perm)

_permutedims(x::AbstractVector, perm) = x
_permutedims(x::Transpose, perm) = perm == (2, 1) ? transpose(x) : x

function _permutedims(x::PermutedDimsArray{<:Any,<:Any,perm,iperm}, new_perm) where {perm,iperm}
    new_perm == iperm && return x.parent
    return PermutedDimsArray(x.parent, perm[[new_perm...]])
end

function write_tensors(io::IO, tensors::OrderedDict{String,AbstractArray})
    for tensor in values(tensors)
        write(io, _permutedims(tensor, Tuple(ndims(tensor):-1:1)))
    end
end
