acc(ŷ, y) = (mode.(ŷ) .== y) |> mean

function learn_curve(model, X, y, meas=accuracy; rng=545)
    training_losses = []
    valid_losses = []
    
    #split training data into training and holdout
    train, valid = partition(eachindex(y), 0.8, shuffle=true, rng=rng)
    data_schedule = range(10, size(train)[1]; step=1)
    
    #iterate over dataset size
    for d in data_schedule
        mach = machine(model, X[train,:], y[train])
        fit!(mach, rows=collect(1:d), force=true, verbosity=0)
        #add loss to training_losses
        train_metric = meas(MLJ.predict(mach, X[train[1:d],:]), y[train[1:d]])
        push!(training_losses, train_metric)
        #test against holdout
        valid_metric = meas(MLJ.predict(mach, X[valid,:]), y[valid])
        push!(valid_losses, valid_metric)
        @show d, train_metric, valid_metric
    end
    
    return data_schedule, training_losses, valid_losses    
end