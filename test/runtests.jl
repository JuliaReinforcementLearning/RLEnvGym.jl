using ReinforcementLearningEnvironmentGym
using PyCall
using Test

for x in ["CartPole-v0"]
    env = GymEnv(x)
    @test typeof(reset!(env)) == NamedTuple{(:observation,), Tuple{PyArray{Float64, 1}}}
    @test typeof(interact!(env, 1)) == NamedTuple{(:observation, :reward, :isdone), Tuple{Array{Float64, 1}, Float64, Bool}}
end
