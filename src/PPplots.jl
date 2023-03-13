# Recipe for plotting PointSets
@recipe function f(s::PointSet, b::Box)
    y = map(u -> (coordinates(u)[1],coordinates(u)[2]), s.items)
    lo, up = coordinates.(extrema(b))
    # Settings
    seriestype --> :scatter  
    legend --> false         
    markercolor --> :black
    xlim --> (lo[1],up[1])
    ylim --> (lo[2],up[2])
    aspectratio --> 1.0
    markersize --> 5
    return y
end


function CondIntPlot(logλ::Function,s::PointSet,W::Box; N=50)
    λ(u,s) = exp(logλ(u,s))
    lo, up = coordinates.(extrema(W))
    u1 = range(lo[1],stop=up[1],length=N)
    u2 = range(lo[2],stop=up[2],length=N)
    z = [λ(Point([x,y]),s) for x in u1, y in u2]
    plt = heatmap(u1,u2,z', color=cgrad([:white,:red]))
    plot!(plt,s,W, mc=:black)
    return plt
end