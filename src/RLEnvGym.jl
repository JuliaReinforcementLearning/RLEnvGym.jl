module RLEnvGym
using Reexport
@reexport using ReinforcementLearning
import ReinforcementLearning:interact!, reset!, getstate, plotenv
using PyCall
@pyimport gym
# @pyimport roboschool

function getspace(space)
    if pyisinstance(space, gym.spaces[:box][:Box])
        ReinforcementLearning.Box(space[:low], space[:high])
    elseif pyisinstance(space, gym.spaces[:discrete][:Discrete])
        1:space[:n]
    else
        error("Don't know how to convert $(pytypeof(space)).")
    end
end
mutable struct GymEnvState
    done::Bool
end
struct GymEnv{TObject, TObsSpace, TActionSpace}
    pyobj::TObject
    observation_space::TObsSpace
    action_space::TActionSpace
    state::GymEnvState
end
function GymEnv(name::String)
    pyenv = gym.make(name)
    obsspace = getspace(pyenv[:observation_space])
    actspace = getspace(pyenv[:action_space])
    env = GymEnv(pyenv, obsspace, actspace, GymEnvState(false))
    reset!(env)
    env
end

function interactgym!(action, env)
    if env.state.done 
        s = reset!(env)
        r = 0
        d = false
    else
        s, r, d = env.pyobj[:step](action)
    end
    env.state.done = d
    s, r, d
end
interact!(action, env::GymEnv) = interactgym!(action, env)
interact!(action::Int64, env::GymEnv) = interactgym!(action - 1, env)
reset!(env::GymEnv) = env.pyobj[:reset]()
getstate(env::GymEnv) = (Float64[env.pyobj[:env][:state]...], false) # doesn't work for all envs

plotenv(env::GymEnv, s, a, r, d) = env.pyobj[:render]()
listallenvs() = gym.envs[:registry][:all]()

export GymEnv, listallenvs

end # module
