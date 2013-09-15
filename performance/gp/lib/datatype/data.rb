module GP
  class Data
    def initialize
      @options = []
    end
    def <<(option)
      @options << option
    end
    def option_expr
      return @options.join ' '
    end
  end
end
