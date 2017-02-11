def mixed_fraction(s)

  flag = 0
	buff = s.index('/')
	num = s[0...buff].to_f
	div = s[buff+1..s.length].to_f
	full_num = (num/div).to_f
	if full_num >= 0
		buff = full_num.divmod 1
	else
		buff = full_num.divmod -1
    flag = 1
		buff[0] *= -1
		buff[1] *= -1
	end

	if buff[1] == 0
		return buff[0].to_s
	elsif buff[0] == 0
    if flag == 1
      buff[1] *= -1
    end
		return buff[1].rationalize.to_s
	else		
		return buff[0].to_s + ' ' + buff[1].rationalize(0.001).to_s
	end

end