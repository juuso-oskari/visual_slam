from graphslam.load import load_g2o_se3

if __name__=="__main__":
    g = load_g2o_se3("after_opt.g2o")
    g.plot(vertex_markersize=1)