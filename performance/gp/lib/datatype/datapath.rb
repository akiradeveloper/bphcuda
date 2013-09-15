module GP
  class << self
    class DataPath < GP::Data
      def initialize(expr, &proc)
        super()
        @expr = expr
        yield self
      end
      def data_expr
        return @expr
      end
      def inline_data
        return ''
      end
    end
  end
end
