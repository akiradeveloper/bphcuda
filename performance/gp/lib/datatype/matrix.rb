module GP
  class Matrix < GP::Data
    def self.accept?(expr)
      return expr.kind_of? ::Matrix
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
      (0...@expr.row_size).each do |i|
        row = []
        (0...@expr.column_size).each do |j|
          row << @expr[i, j] 
        end
        s << row.join(' ')
        s << "\n"
      end
      s
    end
  end
end
