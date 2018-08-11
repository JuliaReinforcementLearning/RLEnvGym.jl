__precompile__(false)
module RLEnvGym
using Reexport
@reexport using ReinforcementLearning
import ReinforcementLearning:interact!, reset!, getstate, plotenv
using PyCall
@pyimport gym

function getspace(space)
    if pyisinstance(space, gym.spaces[:box][:Box])
        ReinforcementLearning.Box(space[:low], space[:high])
    elseif pyisinstance(space, gym.spaces[:discrete][:Discrete])
        1:space[:n]
    else
        error("Don't know how to convert $(pytypeof(space)).")
    end
end
struct GymEnv{TObject, TObsSpace, TActionSpace}
    pyobj::TObject
    observation_space::TObsSpace
    action_space::TActionSpace
    state::PyObject
end
function GymEnv(name::String)
    pyenv = gym.make(name)
    obsspace = getspace(pyenv[:observation_space])
    actspace = getspace(pyenv[:action_space])
    pyenv[:reset]()
    state = PyNULL()
    pycall!(state, pyenv[:step], PyVector, pyenv[:action_space][:sample]())
    pyenv[:reset]()
    GymEnv(pyenv, obsspace, actspace, state)
end

function interactgym!(action, env)
    if env.state[3]
        reset!(env)
    end
    pycall!(env.state, env.pyobj[:step], PyVector, action)
    env.state[1], env.state[2], env.state[3]
end
interact!(action, env::GymEnv) = interactgym!(action, env)
interact!(action::Int64, env::GymEnv) = interactgym!(action - 1, env)
reset!(env::GymEnv) = env.pyobj[:reset]()
getstate(env::GymEnv) = (Float64[env.pyobj[:env][:state]...], false) # doesn't work for all envs

plotenv(env::GymEnv, s, a, r, d) = env.pyobj[:render]()
listallenvs() = gym.envs[:registry][:all]()

export GymEnv, listallenvs

end # module
