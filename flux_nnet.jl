# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X, y, Xtest, ytest) = (data["X"], data["y"], data["Xtest"], data["ytest"])
y[y.==2] .= -1
ytest[ytest.==2] .= -1
(n, d) = size(X)

include("logreg.jl")
model = logReg12(X, y)

using Flux
# implementing a neural network with 2 hidden layers, 
# using the Dense function within the Chain function
n_units = 128
W1 = randn(n_units, d)
W2 = randn(n_units, n_units)

v = randn(n_units + d)
v[end-d+1:end] = model.w

vt = reshape(v, 1, n_units + d)

maxIter = 10000
batch_size = 10
stepSize(t) = 1 / sqrt(t)

l1 = Dense(W1, true, tanh)
l2 = Dense(W2, true, tanh)
ol = Dense(vt, true, identity)

model = Chain(SkipConnection(Chain(l1,l2), (mx, x) -> [mx; x]),ol)
penalty() = 0.01*(
    sum(abs2, l1.weight)+sum(abs2,l1.bias)+
    sum(abs2, l2.weight)+sum(abs2,l2.bias)+
    sum(abs2, ol.weight)+sum(abs2,ol.bias)
)

loss(x, y) = log(1 + exp(-y * model(x)[1])) + penalty()

for t in 1:maxIter

    g = Array{Any}(undef, batch_size)
    for iter in 1:batch_size
        i = rand(1:n)
        f_layer = loss(X[i, :], y[i])
        g[iter] = gradient(Flux.params(model)) do
            loss(X[i, :], y[i])
        end
    end

    # For loop over layers
    for p in Flux.params(model)
        g_avg = g[1][p]
        for iter in 2:batch_size
            #g_avg .+= g[iter][p]
            g_avg = g_avg + g[iter][p]
        end
        g_avg ./= batch_size
        Flux.update!(p, stepSize(t) * g_avg)
    end
end



