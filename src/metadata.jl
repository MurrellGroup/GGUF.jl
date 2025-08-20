@enum ValueType::UInt32 begin
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12
end

const VALUE_TYPE_TO_TYPE = Dict(
    UINT8 => UInt8,
    INT8 => Int8,
    UINT16 => UInt16,
    INT16 => Int16,
    UINT32 => UInt32,
    INT32 => Int32,
    FLOAT32 => Float32,
    BOOL => Bool,
    STRING => String,
    ARRAY => Array,
    UINT64 => UInt64,
    INT64 => Int64,
    FLOAT64 => Float64,
)

const TYPE_TO_VALUE_TYPE = Dict(reverse(kv) for kv in VALUE_TYPE_TO_TYPE)

function value_type_to_type(value_type::ValueType)
    @something get(VALUE_TYPE_TO_TYPE, value_type, nothing) error("Unsupported value type: $value_type")
end

function type_to_value_type(type)
    if type <: AbstractString
        return STRING
    elseif type <: AbstractArray
        return ARRAY
    else
        return @something get(TYPE_TO_VALUE_TYPE, type, nothing) error("Unsupported type: $type")
    end
end

function read_metadata_value(io::IO, value_type::ValueType)
    type = value_type_to_type(value_type)
    if isbitstype(type)
        return read(io, type)
    elseif type == String
        return read_string(io)
    elseif type == Array
        elem_value_type = ValueType(read(io, UInt32))
        elem_type = VALUE_TYPE_TO_TYPE[elem_value_type]
        count = read(io, UInt64)
        result = Array{elem_type}(undef, count)
        for i in 1:count
            result[i] = read_metadata_value(io, elem_value_type)
        end
        return result
    else
        throw(ArgumentError("Unsupported GGUF metadata value type: $(value_type)"))
    end
end

function read_metadata(io::IO, count::Integer)
    metadata = OrderedDict{String,Any}()
    for _ in 1:count
        key = read_string(io)
        vtype = ValueType(read(io, UInt32))
        value = read_metadata_value(io, vtype)
        metadata[key] = value
    end
    return metadata
end

function write_metadata_value(io::IO, value)
    if value isa AbstractString
        write_string(io, value)
    elseif value isa AbstractArray
        write(io, UInt32(type_to_value_type(eltype(value))))
        write(io, UInt64(length(value)))
        for elem in value
            write_metadata_value(io, elem)
        end
    else
        isbitstype(typeof(value)) || throw(ArgumentError("Unsupported metadata value type: $(typeof(value))"))
        write(io, value)
    end
end

function write_metadata(io::IO, metadata::OrderedDict{String,Any})
    for (key, value) in metadata
        write_string(io, key)
        write(io, UInt32(type_to_value_type(typeof(value))))
        write_metadata_value(io, value)
    end
end
