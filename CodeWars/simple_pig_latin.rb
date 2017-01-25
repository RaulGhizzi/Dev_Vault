def pig_it text
  
  words = text.split
  buff = []
  output = ''
  for var in words do
    if var.match(/[[:alpha:]]/) then
    	buff = var[1..var.length] + var[0] + 'ay' + ' ' 
  		output = output + buff
    else
    	output = output + var + ' '
    end
  	
  end

  output.chop!
  return output
end




