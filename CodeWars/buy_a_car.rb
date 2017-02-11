def nbMonths(startPriceOld, startPriceNew, savingperMonth, percentLossByMonth)
    months = 0
    price_old = startPriceOld
    price_new = startPriceNew
    degrading = percentLossByMonth
    man_cash = price_old
    flag = TRUE
    
    if man_cash >= price_new
        flag = FALSE
    end
          
    while flag do
      months += 1
      if months%2 == 0
        degrading += 0.5
      end
      
      price_old -= price_old*(degrading/100)
      price_new -= price_new*(degrading/100)
      man_cash = price_old + savingperMonth*months 
      if man_cash >= price_new
        flag = FALSE
      end
      
    end
    rest = man_cash - price_new
    return months, rest.round
    
end