module GP
  class Array2 < GP::Data
    def self.accept?(expr)
      unless expr.kind_of? Array 
        return false
      end        
      unless expr.size() == 2
        return false
      end
      return true
    end
    def initialize(expr, &proc)
      super()
      @expr = expr
      yield self
    end
    def data_expr
      return '"-"'
    end
    def inline_data
      s = ''
      xs = @expr[0]
      ys = @expr[1]
      unless xs.size == ys.size
        raise "size of arrays must be equal"
      end
      size = xs.size
      (0...size).each do |i|
        s << "#{xs[i]} #{ys[i]}\n" 
      end
      s
    end
  end
end
