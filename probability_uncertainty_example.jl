using Flux, Plots, Statistics, Distributions
using LaTeXStrings

x = range(-2, stop = 3, length = 50)
y = hcat(2.5 .* sin.(collect(x)), 2.5 .* cos.(collect(x)))
soft = [softmax(y[i, :])[1] for i=1:size(y,1)]

y_draws = Array{Float64}(50, 2, 1000)
for i in 1:1000
    #e = rand(Normal(0,1))
    y_draws[:, :, i] = hcat(2.5 .* sin.(collect(x)) .+ rand(Normal(0,1)) , 2.5 .* cos.(collect(x)) .+ rand(Normal(0,1)))
end
soft_draws = Array{Float64}(50, 1000)
for i in 1:50
    for j in 1:1000
        soft_draws[i, j] = softmax(y_draws[i, :, j])[1]
    end
end


soft_5 = [quantile(soft_draws[i, :], .025) for i=1:size(soft_draws, 1)]
soft_95 = [quantile(soft_draws[i, :], .975) for i=1:size(soft_draws, 1)]
soft_mean = [mean(soft_draws[i, :]) for i=1:size(soft_draws, 1)]

y_5 = hcat([quantile(y_draws[i, 1, :], .025) for i=1:size(y_draws, 1)],
        [quantile(y_draws[i, 2, :], .025) for i=1:size(y_draws, 1)])
y_95 = hcat([quantile(y_draws[i, 1, :], .975) for i=1:size(y_draws, 1)],
        [quantile(y_draws[i, 2, :], .975) for i=1:size(y_draws, 1)])
y_mean = hcat([mean(y_draws[i, 1, :]) for i=1:size(y_draws, 1)],
        [mean(y_draws[i, 2, :]) for i=1:size(y_draws, 1)])

plot(x, y[:, 1], label = L"\sigma(f(x))", legend=:bottomright,
    ribbon = (y[:, 1]-y_5[:, 1], y_95[:, 1]-y[:, 1]),
    color = :black,
    fillcolor = :grey)

plot(x, soft, label = L"\sigma(f(x))", legend=:bottomright,
    color = :red,
    fillcolor = :red)
savefig("uncertainty_example_post_softmax_slide1.png")

plot(x, soft, label = L"\sigma(f(x))", legend=:bottomright,
    ribbon = (soft-soft_5, soft_95-soft),
    color = :red,
    fillcolor = :red)
savefig("uncertainty_example_post_softmax_slide2.png")


plot(x, y[:, 1], label = L"f(x)", legend=:bottomright,
    color = :red,
    fillcolor = :red)
plot!(x, y[:, 2], label = L"\tilde{f}(x)", legend=:bottomright,
    color = :blue,
    fillcolor = :blue)
savefig("uncertainty_example_pre_softmax_slide1")

plot(x, y[:, 1], label = L"\sigma(f(x))", legend=:bottomright,
    color = :red,
    fillcolor = :red,
    ribbon = (y[:, 1]-y_5[:, 1], y_95[:, 1]-y[:, 1]))
plot!(x, y[:, 2], label = L"\tilde{f}(x)", legend=:bottomright,
        color = :blue,
        fillcolor = :blue,
        ribbon = (y[:, 2]-y_5[:, 2], y_95[:, 2]-y[:, 2]))
savefig("uncertainty_example_pre_softmax_slide2")
