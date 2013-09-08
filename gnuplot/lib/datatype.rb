["data", "array2", "array3", "datapath", "mathexpr", "matrix"].each do |f|
  require ["datatype", f].join "/"
end

module GP
  def eval(expr, &proc)
    # String is either expression or file path
    if expr.kind_of? ::String
      if ::File.file?(expr)
        return DataPath.new(expr, &proc)
      end
      # arithmatic expression
      return MathExpr.new(expr, &proc)
    end
    if expr.kind_of? ::Array
      return Array2.new(expr, &proc) if Array2.accept?(expr)
      return Array3.new(expr, &proc) if Array3.accept?(expr)
    end
    if expr.kind_of? ::Matrix
      return GP::Matrix.new(expr, &proc) if GP::Matrix.accept?(expr)
    end
    raise "#{expr.class} is not a supported data expression"
  end
end
