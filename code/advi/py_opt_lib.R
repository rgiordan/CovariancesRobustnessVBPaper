library(reticulate)

InitializePython <- function() {
  use_python("/usr/bin/python3")
  py_run_string("import numpy as np")
  py_run_string("import scipy as sp")
  py_run_string("from scipy import optimize")


  reticulate::py_run_string("
class OptObjectiveFuns(object):
  def __init__(self, Objective, ObjectiveGrad, ObjectiveHess, ObjectiveHVP):
    self.Objective = Objective
    self.ObjectiveGrad = ObjectiveGrad
    self.ObjectiveHess = ObjectiveHess
    self.ObjectiveHVP = ObjectiveHVP

  def OptObjective(self, x):
    return self.Objective(x)

  def OptObjectiveGrad(self, x):
    obj_grad = self.ObjectiveGrad(x)
    return np.squeeze(obj_grad)

  def OptObjectiveHess(self, x):
    obj_hess = self.ObjectiveHess(x)
    return np.squeeze(obj_hess)

  def OptObjectiveHVP(self, x, vec):
    obj_hvp = self.ObjectiveHVP(x, vec)
    return np.squeeze(obj_hvp)
")

  return(reticulate::import_main())
}


DefinePythonOptimizationFunctions <- function(
  name, Objective, ObjectiveGrad=NULL, ObjectiveHess=NULL, ObjectiveHVP=NULL) {
  py_main <- reticulate::import_main()
  py_main[[name]] <- py_main$OptObjectiveFuns(Objective, ObjectiveGrad, ObjectiveHess, ObjectiveHVP)
  return(py_main[[name]])
}


PyPrint <- function(py_expression) {
  py_main <- reticulate::import_main()
  py_run_string(sprintf("pyprintoutput = %s", py_expression))
  print(py_main$pyprintoutput)
}


PythonOptimizeTrustNCG <- function(init_x, opt_obj_name, maxiter=200, gtol=1e-6) {
  `%_%` <- function(x, y) { paste(x, y, sep="")}

  py_main$init_x <- init_x
  reticulate::py_run_string("
sp_opt = sp.optimize.minimize(
    fun=" %_% opt_obj_name %_% ".Objective,
    x0=init_x,
    jac=" %_% opt_obj_name %_% ".OptObjectiveGrad,
    hessp=" %_% opt_obj_name %_% ".OptObjectiveHVP,
    method='trust-ncg',
    options={'maxiter': " %_%  maxiter %_% ",
             'gtol':" %_% gtol %_% ", 'disp': True})
")
  message <- PyPrint("sp_opt.message")
  return(list(x=py_main$sp_opt$x, message=message))
}





if (FALSE) {
  # Tests
  py_main <- InitializePython()

  d <- 5
  a_mat <- diag(d) + matrix(0.1, nrow=d, ncol=d)
  x <- runif(d)
  Objective <- function(x) {
    as.numeric(0.5 * t(x) %*% a_mat %*% x)
  }

  ObjectiveGrad <- function(x) {
    return(a_mat %*% x)
  }

  ObjectiveHess <- function(x) {
    return(a_mat)
  }

  ObjectiveHVP <- function(x, vec) {
    return(a_mat %*% vec)
  }

  py_main$d <- as.integer(d)
  py_main$init_x <- py_main$x <- np_array(x)
  PyPrint("init_x.shape")

  DefinePythonOptimizationFunctions(
    "opt_obj", Objective, ObjectiveGrad, ObjectiveHess, ObjectiveHVP)


  reticulate::py_run_string("
x0 = np.random.random(d)
y = opt_obj.OptObjective(x0)
")

  reticulate::py_run_string("
sp_opt = sp.optimize.minimize(
    fun=opt_obj.OptObjective,
    jac=opt_obj.OptObjectiveGrad,
    x0=init_x)
")
  PyPrint("sp_opt")


  reticulate::py_run_string("
sp_opt = sp.optimize.minimize(
    fun=opt_obj.Objective,
    x0=init_x,
    jac=opt_obj.OptObjectiveGrad,
    hessp=opt_obj.OptObjectiveHVP,
    method='trust-ncg',
    options={'maxiter': 200, 'gtol': 1e-6, 'disp': True})
")
  PyPrint("sp_opt")

}
