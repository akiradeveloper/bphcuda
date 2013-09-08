require "gp"

if __FILE__ == $0
  GP.open() do |gp|
    gp.set('xrange', '[0:5]')
    gp.plot do |plot|
      plot << GP.eval([[0,1], [1,2]]) do |d|
        d << 'with lines'
        d << 'title "unko"'
      end
    end 
  end  
end
