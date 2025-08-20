using GGUF
using Test

@testset "GGUF.jl" begin

    mktempdir() do dir
        metadata = Dict("general.name" => "Dummy", "general.tags" => ["test", "dummy"])
        tensors = Dict(
            "proj" => reshape(collect(1.0:600.0), 20, 30),
            "bias" => collect(Int32, 1:4000),
            "embed" => reshape(collect(Float16, 1:40000), 20, 50, 40),
        )
        gguf = GGUFObject(metadata, tensors)
        path = joinpath(dir, "test.gguf")
        GGUF.serialize(path, gguf)
        new_gguf = GGUF.deserialize(path)
        @testset "metadata" for ((k1, v1), (k2, v2)) in zip(new_gguf.metadata, gguf.metadata)
            @test k1 == k2
            @test v1 == v2
        end
        @testset "tensors" for ((k1, v1), (k2, v2)) in zip(new_gguf.tensors, gguf.tensors)
            @test k1 == k2
            @test vec(v1) == vec(v2)
            @test size(v1) == size(v2)
            @test v1 == v2
        end
    end

end
