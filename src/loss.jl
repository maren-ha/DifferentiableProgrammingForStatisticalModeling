"""
loss function 
"""

function loss(y, z, r, beta)
    sqerror = 0.0 # squared error
    firstseen = false # set to true after skipping potential missings 
    last_y = 0.0 # prevalent cases from previous time point
    contribno = 0.0 # number of non-missing observations

    for t = 1:length(y)
        # skip missings at the start until first reported value
        if !firstseen 
            if r[t] == 1
                firstseen = true
                last_y = y[t]
            else
                continue
            end
        else # make a prediction for the current increment
            pred_dy = beta[1] + beta[2] * last_y + beta[3] * z[t-1] 
            if r[t] == 1
                dy = y[t] - last_y
                sqerror += (dy - pred_dy)^2
                contribno += 1.0
                last_y = y[t]
            else
                last_y += pred_dy
            end 
        end
    end
    return sqerror/contribno # return MSE over all reported time points end
end

function loss_locflinreg(dy, y, z, beta)
    dy_hat = beta[1] .+ y .* beta[2] .+ z .* beta[3]
    val = sum((dy_hat .- dy).^2)/length(y)
    return val 
end 
