acc(ŷ, y) = (mode.(ŷ) .== y) |> mean

function learn_curve(model, X, y, meas=accuracy; rng=545, step=5)
    training_losses = []
    valid_losses = []
    
    #split training data into training and holdout
    train, valid = partition(eachindex(y), 0.8, shuffle=true, rng=rng)
    data_schedule = range(10, size(train)[1]; step=step)
    
    #iterate over dataset size
    for d in data_schedule
        mach = machine(model, X[train,:], y[train])
        m = fit!(mach, rows=collect(1:d), force=true, verbosity=0)
        #add loss to training_losses
        train_metric = meas(MLJ.predict(m, X[train[1:d],:]), y[train[1:d]])
        push!(training_losses, train_metric)
        #test against holdout
        valid_metric = meas(MLJ.predict(m, X[valid,:]), y[valid])
        # @show MLJ.predict(m, X[valid,:]), y[valid]
        push!(valid_losses, valid_metric)
        @show d, train_metric, valid_metric
    end
    
    return data_schedule, training_losses, valid_losses    
end

function separate_bases(df)
    d = map(s->[i for i in strip(s)], df[:,3])
    a = zeros(size(d)[1], size(d[1])[1])
    # a = Array{Char,2}(undef,(size(d)[1], size(d[1])[1]))

    
    for i in 1:size(a)[1]
        a[i,:] = d[i]
    end
    
    return DataFrame(a)
end