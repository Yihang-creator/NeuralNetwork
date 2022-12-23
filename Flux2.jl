# using JLD
# data = load("mnist35.jld")
# (X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
# y[y.==2] .= -1
# ytest[ytest.==2] .= -1
# (n,d) = size(X)

# using Flux
# model = Chain(Dense(d => 6, tanh), Dense(6 => 3, tanh), Dense(3 => 1, identity))
# loss(x, y) = (1/2)*(model(x)[1]-y)^2
# max_iter = 1000
# stepSize = 1e-4
# for t in 1:max_iter
#     i = rand(1:n)
#     f_layer = loss(X[i,:],y[i])
#     g_layer = gradient(Flux.params(model)) do 
#         loss(X[i,:],y[i])
#     end

#     for p in Flux.params(model)
#         p - stepSize*g_layer[p]
#     end
# end

# s = 0
# ntest = size(Xtest,1)
# for i in 1:ntest
#     yh = sign.(model(Xtest[i,:]))
#     global s += (yh[1] != ytest[i])
# end

# errorRate = s/size(Xtest,1)
# @printf("error rate = %.4f\n",errorRate)

layer = Conv((5,5),3=>6,relu)
Flux.params(layer)