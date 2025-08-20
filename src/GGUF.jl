module GGUF

using BFloat16s: BFloat16
using LinearAlgebra: Transpose
using Mmap: Mmap
using OrderedCollections: OrderedDict

export GGUFObject

function check_magic(io::IO)
    String(read(io, 4)) == "GGUF" || throw(ArgumentError("Not a GGUF file: bad magic"))
end

function check_version(io::IO)
    version = Int(read(io, UInt32))
    version == 3 || throw(ArgumentError("Unsupported GGUF version: $version"))
end

read_string(io::IO) = String(read(io, read(io, UInt64)))
write_string(io::IO, str::AbstractString) = (write(io, UInt64(ncodeunits(str))); write(io, str))

align_up(value::Integer, alignment::Integer) =
    value + (alignment - (value % alignment)) % alignment

include("metadata.jl")
include("tensors.jl")

struct GGUFObject
    metadata::OrderedDict{String,Any}
    tensors::OrderedDict{String,AbstractArray}

    function GGUFObject(metadata, tensors)
        metadata = OrderedDict(k => v for (k, v) in metadata)
        tensors = OrderedDict(k => v for (k, v) in tensors)
        new(metadata, tensors)
    end
end

function Base.show(io::IO, gguf::GGUFObject)
    println(io, summary(gguf), ':')
    println(io, "  metadata entries: $(length(gguf.metadata))")
    print(io, "  tensors: $(length(gguf.tensors))")
end

function deserialize(buf::Vector{UInt8})
    io = IOBuffer(buf)

    check_magic(io)
    check_version(io)

    tensor_count = read(io, UInt64)
    metadata_kv_count = read(io, UInt64)

    metadata = read_metadata(io, metadata_kv_count)
    tensor_info_dict = read_tensor_info_dict(io, tensor_count)

    alignment = get(metadata, "general.alignment", 32)
    base_offset = align_up(position(io), alignment)

    tensor_data = @view buf[base_offset+1:end]
    tensors = get_tensors(tensor_data, tensor_info_dict)

    return GGUFObject(metadata, tensors)
end

function deserialize(path::AbstractString; mmap=true)
    buf = mmap ? open(Mmap.mmap, path) : read(path)
    return deserialize(buf)
end

function serialize(io::IO, gguf::GGUFObject)
    write(io, "GGUF")
    write(io, UInt32(3))

    write(io, UInt64(length(gguf.tensors)))
    write(io, UInt64(length(gguf.metadata)))

    write_metadata(io, gguf.metadata)

    write_tensor_info_dict(io, get_tensor_info_dict(gguf.tensors))
    
    alignment = get(gguf.metadata, "general.alignment", 32)
    base_offset = align_up(position(io), alignment)
    seek(io, base_offset)

    write_tensors(io, gguf.tensors)

    return nothing
end

function serialize(path::AbstractString, gguf::GGUFObject)
    open(path, "w") do io
        serialize(io, gguf)
    end
end

end
