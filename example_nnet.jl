using Printf

# Load X and y variable
using JLD
using Plots
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
(n,d) = size(X)

@show n,d

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [3 3 2]
nParams = NeuralNet_nParams(d,nHidden)
using Distributions
w = rand(Normal(0, 1.5),(nParams,1))

# Train with stochastic gradient
maxIter = 10000
stepSize = 1e-3
batchSize = 10
for t in 1:maxIter
	alpha = stepSize
	# The stochastic gradient update:
	g0 = zeros(size(w,1))
	for i in 1:batchSize
		(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
		g0 += g
	end
	global w = w - alpha*g0


	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		@printf("Training iteration = %d\n",t-1)
		xVals = -10:.05:10
		Xhat = zeros(length(xVals),1)
		Xhat[:] .= xVals
		yhat = NeuralNet_predict(w,Xhat,nHidden)
		scatter(X,y,legend=false,linestyle=:dot)
		plot!(Xhat,yhat,legend=false)
		gui()
		sleep(.1)
	end
end
plot!()
