#include "common/BasicTypes.h"
#include "solver_core/SolverInterface.h"
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include "common/MatlabHelper.h"

namespace py = pybind11;
using namespace CRISP;

// define the module
PYBIND11_MODULE(pyCRISP, m) {
  m.doc() = "python interface for ContactSolver";  // optional module docstring
  py::enum_<CppAdInterface::ModelInfoLevel>(m, "ModelInfoLevel")
      .value("FIRST_ORDER", CppAdInterface::ModelInfoLevel::FIRST_ORDER)
      .value("SECOND_ORDER", CppAdInterface::ModelInfoLevel::SECOND_ORDER)
      .export_values();

  // Register the ObjectiveFunction SpecifiedFunctionLevel enum with a unique name
  py::enum_<ObjectiveFunction::SpecifiedFunctionLevel>(m, "ObjectiveFunction_SpecifiedFunctionLevel")
      .value("NONE", ObjectiveFunction::SpecifiedFunctionLevel::NONE)
      .export_values();

  // Register the ConstraintFunction SpecifiedFunctionLevel enum with a unique name
  py::enum_<ConstraintFunction::SpecifiedFunctionLevel>(m, "ConstraintFunction_SpecifiedFunctionLevel")
      .value("NONE", ConstraintFunction::SpecifiedFunctionLevel::NONE)
      .export_values();
  // expose the solver interface
  py::class_<SolverInterface>(m, "SolverInterface")
      .def(py::init<OptimizationProblem&, SolverParameters&>())
      .def("initialize", &SolverInterface::initialize)
      .def("reset_problem", &SolverInterface::resetProblem)  // reset problem with new initial guess
      .def("set_problem_parameters",
           &SolverInterface::setProblemParameters)  // problem related data, related to your obj, constraints, like the
                                                    // tracking reference, terminal states, etc
      .def("set_hyper_parameters", &SolverInterface::setHyperParameters)  // hyperparameters for the solver, like max
                                                                          // iterations, trust region radius, etc
      .def("solve", &SolverInterface::solve)
      .def("get_solution", &SolverInterface::getSolution);
  // .def("save_results", &SolverInterface::saveResults);

  // expose optimization problem
  py::class_<OptimizationProblem>(m, "OptimizationProblem")
      .def(py::init<size_t, const std::string&>())
      .def("add_objective", &OptimizationProblem::addObjective)
      .def("add_equality_constraint", &OptimizationProblem::addEqualityConstraint)
      .def("add_inequality_constraint", &OptimizationProblem::addInequalityConstraint)
      .def("set_parameters", &OptimizationProblem::setParameters)
      .def("parse_drake_mathematical_program", &OptimizationProblem::parseDrakeMathematicalProgram);

  // expose solver parameters
  py::class_<SolverParameters>(m, "SolverParameters")
      .def(py::init<>())
      .def("set_parameters", &SolverParameters::setParameters)
      .def("get_parameters", &SolverParameters::getParameters);

  // expose matlab helper
  // py::class_<MatlabHelper>(m, "MatlabHelper")
  //     .def_static("read_variable_from_mat_file", &MatlabHelper::readVariableFromMatFile)
  //     .def_static("read_variable_from_mat_file_py", &MatlabHelper::readVariableFromMatFilePy);

  // expose ObjectiveFunction
  py::class_<ObjectiveFunction, std::shared_ptr<ObjectiveFunction>>(m, "ObjectiveFunction")
      .def(py::init<size_t, const std::string&, const std::string&, const std::string&, bool,
                    CppAdInterface::ModelInfoLevel, ObjectiveFunction::SpecifiedFunctionLevel>(),
           py::arg("variableDim"), py::arg("modelName"), py::arg("folderName"), py::arg("functionName"),
           py::arg("regenerateLibrary") = false, py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
           py::arg("specifiedFunctionLevel") = ObjectiveFunction::SpecifiedFunctionLevel::NONE)

      .def(py::init<size_t, size_t, const std::string&, const std::string&, const std::string&, bool,
                    CppAdInterface::ModelInfoLevel, ObjectiveFunction::SpecifiedFunctionLevel>(),
           py::arg("variableDim"), py::arg("parameterDim"), py::arg("modelName"), py::arg("folderName"),
           py::arg("functionName"), py::arg("regenerateLibrary") = false,
           py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
           py::arg("specifiedFunctionLevel") = ObjectiveFunction::SpecifiedFunctionLevel::NONE);

  // expose ConstraintFunction
  py::class_<ConstraintFunction, std::shared_ptr<ConstraintFunction>>(m, "ConstraintFunction")
      .def(py::init<size_t, const std::string&, const std::string&, const std::string&, bool,
                    CppAdInterface::ModelInfoLevel, ConstraintFunction::SpecifiedFunctionLevel>(),
           py::arg("variableDim"), py::arg("modelName"), py::arg("folderName"), py::arg("functionName"),
           py::arg("regenerateLibrary") = false, py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
           py::arg("specifiedFunctionLevel") = ConstraintFunction::SpecifiedFunctionLevel::NONE)

      .def(py::init<size_t, size_t, const std::string&, const std::string&, const std::string&, bool,
                    CppAdInterface::ModelInfoLevel, ConstraintFunction::SpecifiedFunctionLevel>(),
           py::arg("variableDim"), py::arg("parameterDim"), py::arg("modelName"), py::arg("folderName"),
           py::arg("functionName"), py::arg("regenerateLibrary") = false,
           py::arg("infoLevel") = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
           py::arg("specifiedFunctionLevel") = ConstraintFunction::SpecifiedFunctionLevel::NONE);
}