#include "Car5D.hpp"
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>
#include <array>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
//#include "Car5D_cuda.hpp"
#include <random>
#include <macro.hpp>
using namespace helperOC;

Car5D::Car5D(
  const beacls::FloatVec& x,
  const beacls::FloatVec& uRange,
  const beacls::FloatVec& aRange,
  const beacls::FloatVec& dRange): 
  DynSys(4, 1, 2, // # states, # control inputs, # disturbances
  beacls::IntegerVec{0},  // Position dimensions
  beacls::IntegerVec{1}), // velocity dimensions
  uRange(uRange), aRange(aRange), dRange(dRange) {
  if (x.size() != 4) {
    std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
  }

  DynSys::set_x(x);
  DynSys::push_back_xhist(x);

}

Car5D::~Car5D() {
}

bool Car5D::operator==(const Car5D& rhs) const {
  if (this == &rhs) return true;
  else if (!DynSys::operator==(rhs)) return false;
  else if ((uRange.size() != rhs.uRange.size()) || 
    !std::equal(uRange.cbegin(), uRange.cend(), rhs.uRange.cbegin())) {
    return false; //!< Speed control bounds
  }
  else if ((aRange.size() != rhs.aRange.size()) || 
    !std::equal(aRange.cbegin(), aRange.cend(), rhs.aRange.cbegin())) {
    return false; //!< Speed control bounds
  }
  else if ((dRange.size() != rhs.dRange.size()) || 
    !std::equal(dRange.cbegin(), dRange.cend(), rhs.dRange.cbegin())) {
    return false; //!< Disturbance
  }
  else return true;
}

bool Car5D::operator==(const DynSys& rhs) const {
  if (this == &rhs) return true;
  else if (typeid(*this) != typeid(rhs)) return false;
  else return operator==(dynamic_cast<const Car5D&>(rhs));
}

bool Car5D::optCtrl(std::vector<beacls::FloatVec>& uOpts,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>& y_ites,
    const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
    const beacls::IntegerVec& y_sizes,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_UMode_Type uMode) const {
  
  const helperOC::DynSys_UMode_Type modified_uMode = 
    (uMode == helperOC::DynSys_UMode_Default) ? 
    helperOC::DynSys_UMode_Max : uMode;

  const size_t y3_size = y_sizes[3];
  const FLOAT_TYPE* deriv3_ptr = deriv_ptrs[3];
  const size_t deriv3_size = deriv_sizes[3];

  if (y3_size == 0 || deriv3_size == 0|| deriv3_ptr == NULL) {
    return false;
  }
    
  uOpts.resize(get_nu());
  uOpts[0].resize(deriv3_size);

  if ((modified_uMode != helperOC::DynSys_UMode_Max) && 
      (modified_uMode != helperOC::DynSys_UMode_Min)) {
    std::cerr << "Unknown uMode!: " << uMode << std::endl;
    return false;   
  }

  const FLOAT_TYPE u_if_p3_pos = 
      (modified_uMode == helperOC::DynSys_UMode_Max) ? uRange[1] : uRange[0];
  const FLOAT_TYPE u_if_p3_neg =
      (modified_uMode == helperOC::DynSys_UMode_Max) ? uRange[0] : uRange[1];
  std::transform(deriv3_ptr, deriv3_ptr + deriv3_size, uOpts[0].begin(), 
      [u_if_p3_pos, u_if_p3_neg](const auto& p3){ 
      return (p3 >= 0) ? u_if_p3_pos : u_if_p3_neg; });

  return true;
}
bool Car5D::optDstb(std::vector<beacls::FloatVec>& dOpts,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>&,
    const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
    const beacls::IntegerVec&,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_DMode_Type dMode) const {
  const helperOC::DynSys_DMode_Type modified_dMode = 
    (dMode == helperOC::DynSys_DMode_Default) ? 
    helperOC::DynSys_DMode_Min : dMode;

  const FLOAT_TYPE* deriv0_ptr = deriv_ptrs[0];
  const FLOAT_TYPE* deriv1_ptr = deriv_ptrs[1];
  const size_t deriv0_size = deriv_sizes[0];

  if (deriv0_size == 0 || deriv0_ptr == NULL || deriv1_ptr == NULL) 
    return false;

  dOpts.resize(get_nd());
  std::for_each(dOpts.begin(), dOpts.end(), [deriv0_size](auto& rhs) 
    { rhs.resize(deriv0_size); });

  beacls::FloatVec& dOpt0 = dOpts[0];
  beacls::FloatVec& dOpt1 = dOpts[1];

  if ((modified_dMode != helperOC::DynSys_DMode_Max) && 
      (modified_dMode != helperOC::DynSys_DMode_Min)) {
    std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
    return false;
  }

  // Disturbance
  const FLOAT_TYPE d_if_p0_pos = 
    (modified_dMode == helperOC::DynSys_DMode_Max) ? dRange[1] : dRange[0];
  const FLOAT_TYPE d_if_p0_neg =
    (modified_dMode == helperOC::DynSys_DMode_Max) ? dRange[0] : dRange[1];
  std::transform(deriv0_ptr, deriv0_ptr + deriv0_size, dOpts[0].begin(), 
      [d_if_p0_pos, d_if_p0_neg](const auto& p0){ 
      return (p0 >= 0) ? d_if_p0_pos: d_if_p0_neg; });

  // Planning control
  const FLOAT_TYPE a_if_minus_p1_pos = 
    (modified_dMode == helperOC::DynSys_DMode_Max) ? aRange[1] : aRange[0];
  const FLOAT_TYPE a_if_minus_p1_neg =
    (modified_dMode == helperOC::DynSys_DMode_Max) ? aRange[0] : aRange[1];
  std::transform(deriv1_ptr, deriv1_ptr + deriv0_size, dOpts[1].begin(), 
      [a_if_minus_p1_pos, a_if_minus_p1_neg](const auto& p1){ 
      return (-p1 >= 0) ? a_if_minus_p1_pos: a_if_minus_p1_neg; });

  return true;
}

bool Car5D::dynamics_cell_helper(beacls::FloatVec& dx,
    const beacls::FloatVec::const_iterator& x_ite1,
    const beacls::FloatVec::const_iterator& x_ite2,
    const beacls::FloatVec::const_iterator& x_ite3,
    const std::vector<beacls::FloatVec>& us,
    const std::vector<beacls::FloatVec>& ds,
    const size_t x_size,
    const size_t dim) const {
    // dx[0] = x[1] + d[0]
    // dx[1] = g * tan(x[2]) - d[1]            (d[1] is planning control)
    // dx[2] = -d1 * x[2] + x[3]
    // dx[3] = -d0 * x[2] + n0 * u
  beacls::FloatVec& dx_dim = dx;
  const size_t dx_dim_size = us[0].size();
  dx.resize(dx_dim_size);

  bool result = true;

  switch (dim) {
    case 0: { // dx[0] = x[1] + d[0]
      const beacls::FloatVec& ds_0 = ds[0];

      std::transform(ds_0.cbegin(), ds_0.cbegin() + dx_dim_size, 
          x_ite1, dx_dim.begin(), std::plus<FLOAT_TYPE>());
    }
    break;

    case 1: { // dx[1] = g * tan(x[2]) - d[1]
      const beacls::FloatVec& ds_1 = ds[1];
      
      std::transform(x_ite2, x_ite2 + dx_dim_size, 
          ds_1.begin(), dx_dim.begin(), [this](const auto& x2, const auto& ds1){ 
              return g*std::tan(x2) - ds1; });
    }
    break;

    case 2: { // dx[2] = -d1 * x[2] + x[3]
      std::transform(x_ite2, x_ite2 + dx_dim_size, x_ite3, dx_dim.begin(), 
        [this](const auto& x2, const auto& x3) { return -d1 * x2 + x3;  });
    }
    break;

    case 3: { // dx[3] = -d0 * x[2] + n0 * u
      const beacls::FloatVec& us_0 = us[0];     
      std::transform(x_ite2, x_ite2 +dx_dim_size, us_0.cbegin(), dx_dim.begin(), 
          [this](const auto& x2, const auto& u) { return -d0*x2 + n0*u; });
    }
    break;

    default: {
      std::cerr << "Only dimension 1-4 are defined for dynamics of Car5D!" 
          << std::endl;
      result = false;
    }
    break;
  }
  return result;
}

bool Car5D::dynamics(std::vector<beacls::FloatVec >& dxs,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator >& x_ites,
    const std::vector<beacls::FloatVec >& us,
    const std::vector<beacls::FloatVec >& ds,
    const beacls::IntegerVec& x_sizes,
    const size_t dst_target_dim) const {
  static const std::vector<beacls::FloatVec>& 
      dummy_ds{beacls::FloatVec{0}, beacls::FloatVec{0}, beacls::FloatVec{0}};
  const std::vector<beacls::FloatVec>& modified_ds = 
      (ds.empty()) ? dummy_ds : ds;

  const beacls::FloatVec::const_iterator& x_ites1 = x_ites[1];
  const beacls::FloatVec::const_iterator& x_ites2 = x_ites[2];
  const beacls::FloatVec::const_iterator& x_ites3 = x_ites[3];

  bool result = true;
  if (dst_target_dim == std::numeric_limits<size_t>::max()) {
    for (size_t dim = 0; dim < 4; ++dim) {
      result &= dynamics_cell_helper(dxs[dim], x_ites1, x_ites2, x_ites3, us, 
          ds, x_sizes[0], dim);
    }
  }
  else  {
    if (dst_target_dim < x_ites.size()) {
      dynamics_cell_helper(dxs[dst_target_dim], x_ites1, x_ites2, x_ites3, us, 
          modified_ds, x_sizes[0], dst_target_dim);
    }
    else {
      std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim 
          << std::endl;
      result = false;
    }
  }
  return result;
}

// ====================================== GPU ONLY
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool Car5D::optCtrl_cuda(
    std::vector<beacls::UVec>& u_uvecs,
    const FLOAT_TYPE,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& deriv_uvecs,
    const helperOC::DynSys_UMode_Type uMode) const {

  if (x_uvecs.size() < 3 || x_uvecs[2].empty() || deriv_uvecs.size() < 3 || 
      deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || 
      deriv_uvecs[2].empty()) {
    return false;
  }
    
  const helperOC::DynSys_UMode_Type modified_uMode = 
    (uMode == helperOC::DynSys_UMode_Default) ? 
    helperOC::DynSys_UMode_Max : uMode;

  const auto vrange_minmax = 
    beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());

  const FLOAT_TYPE vrange_min = vrange_minmax.first;
  const FLOAT_TYPE vrange_max = vrange_minmax.second;

  return Car5D_CUDA::optCtrl_execute_cuda(u_uvecs, x_uvecs, deriv_uvecs, 
    wMax, vrange_max, vrange_min, modified_uMode);
}

bool Car5D::optDstb_cuda(
    std::vector<beacls::UVec>& d_uvecs,
    const FLOAT_TYPE,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& deriv_uvecs,
    const helperOC::DynSys_DMode_Type dMode) const {

  if (deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty()
      || deriv_uvecs[2].empty()) {
    return false;
  }
  
  const helperOC::DynSys_DMode_Type modified_dMode = 
    (dMode == helperOC::DynSys_DMode_Default) ? 
    helperOC::DynSys_DMode_Min : dMode;

  return Car5D_CUDA::optDstb_execute_cuda(d_uvecs, x_uvecs, deriv_uvecs, dMax,
    modified_dMode);
}

bool Car5D::dynamics_cuda(
    std::vector<beacls::UVec>& dx_uvecs,
    const FLOAT_TYPE,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& u_uvecs,
    const std::vector<beacls::UVec>& d_uvecs,
    const size_t dst_target_dim) const {

  beacls::FloatVec dummy_d_vec{ 0 };
  std::vector<beacls::UVec> dummy_d_uvecs;

  if (d_uvecs.empty()) {
    dummy_d_uvecs.resize(get_nd());
    std::for_each(dummy_d_uvecs.begin(), dummy_d_uvecs.end(), 
      [&dummy_d_vec](auto& rhs) { 
      rhs = beacls::UVec(dummy_d_vec, beacls::UVecType_Vector, false); });
  }

  const std::vector<beacls::UVec>& modified_d_uvecs = 
    (d_uvecs.empty()) ? dummy_d_uvecs : d_uvecs;

  bool result = true;
  if (dst_target_dim == std::numeric_limits<size_t>::max()) {
    result &= Car5D_CUDA::dynamics_cell_helper_execute_cuda_dimAll(
      dx_uvecs, x_uvecs, u_uvecs, modified_d_uvecs);
  }
  else {
    if (dst_target_dim < x_uvecs.size()) {
      Car5D_CUDA::dynamics_cell_helper_execute_cuda(dx_uvecs[dst_target_dim], 
        x_uvecs, u_uvecs, modified_d_uvecs, dst_target_dim);
    }
    else {
      std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim 
        << std::endl;
      result = false;
    }
  }
  return result;
}

bool Car5D::optCtrl_cuda(
    std::vector<beacls::UVec>& uU_uvecs,
    std::vector<beacls::UVec>& uL_uvecs,
    const FLOAT_TYPE,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& derivMax_uvecs,
    const std::vector<beacls::UVec>& derivMin_uvecs,
    const helperOC::DynSys_UMode_Type uMode) const {

  if (x_uvecs.size() < 3 || x_uvecs[2].empty() || derivMax_uvecs.size() < 3 || 
      derivMax_uvecs[0].empty() || derivMax_uvecs[1].empty() || 
      derivMax_uvecs[2].empty()) {
    return false;
  }

  const helperOC::DynSys_UMode_Type modified_uMode = 
    (uMode == helperOC::DynSys_UMode_Default) ? 
    helperOC::DynSys_UMode_Max : uMode;

  const auto vrange_minmax = 
    beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());

  const FLOAT_TYPE vrange_min = vrange_minmax.first;
  const FLOAT_TYPE vrange_max = vrange_minmax.second;

  return Car5D_CUDA::optCtrl_execute_cuda(uU_uvecs, uL_uvecs, x_uvecs, 
    derivMax_uvecs, derivMin_uvecs, wMax, vrange_max, vrange_min, modified_uMode);
}

bool Car5D::optDstb_cuda(
    std::vector<beacls::UVec>& dU_uvecs,
    std::vector<beacls::UVec>& dL_uvecs,
    const FLOAT_TYPE,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& derivMax_uvecs,
    const std::vector<beacls::UVec>& derivMin_uvecs,
    const helperOC::DynSys_DMode_Type dMode) const {

  if (derivMax_uvecs.size() < 3 || derivMax_uvecs[0].empty() || 
      derivMax_uvecs[1].empty() || derivMax_uvecs[2].empty()) {
    return false;
  }

  const helperOC::DynSys_DMode_Type modified_dMode = 
    (dMode == helperOC::DynSys_DMode_Default) ? 
    helperOC::DynSys_DMode_Min : dMode;

  return Car5D_CUDA::optDstb_execute_cuda(dU_uvecs, dL_uvecs, x_uvecs, 
    derivMax_uvecs, derivMin_uvecs, dMax, modified_dMode);
}

bool Car5D::dynamics_cuda(
    beacls::UVec& alpha_uvec,
    const FLOAT_TYPE,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& uU_uvecs,
    const std::vector<beacls::UVec>& uL_uvecs,
    const std::vector<beacls::UVec>& dU_uvecs,
    const std::vector<beacls::UVec>& dL_uvecs,
    const size_t dst_target_dim) const {

  beacls::FloatVec dummy_d_vec{ 0 };
  std::vector<beacls::UVec> dummy_d_uvecs;

  if (dU_uvecs.empty() || dL_uvecs.empty()) {
    dummy_d_uvecs.resize(get_nd());
    std::for_each(dummy_d_uvecs.begin(), dummy_d_uvecs.end(), 
      [&dummy_d_vec](auto& rhs) {
      rhs = beacls::UVec(dummy_d_vec, beacls::UVecType_Vector, false);
    });
  }
  const std::vector<beacls::UVec>& modified_dU_uvecs = 
    (dU_uvecs.empty()) ? dummy_d_uvecs : dU_uvecs;
  const std::vector<beacls::UVec>& modified_dL_uvecs = 
    (dL_uvecs.empty()) ? dummy_d_uvecs : dL_uvecs;

  bool result = true;
  if (dst_target_dim < x_uvecs.size()) {
    Car5D_CUDA::dynamics_cell_helper_execute_cuda(alpha_uvec, x_uvecs, 
      uU_uvecs, uL_uvecs, modified_dU_uvecs, modified_dL_uvecs, dst_target_dim);
  }
  else {
    std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim 
      << std::endl;
    result = false;
  }
  return result;
}

bool Car5D::HamFunction_cuda(
    beacls::UVec& hamValue_uvec,
    const DynSysSchemeData* schemeData,
    const FLOAT_TYPE,
    const beacls::UVec&,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& deriv_uvecs,
    const size_t,
    const size_t,
    const bool negate) const {
  if (x_uvecs.size() < 3 || x_uvecs[2].empty() || deriv_uvecs.size() < 3 || 
    deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty()) {
    return false;
  }

  if (deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() 
    || deriv_uvecs[2].empty()) {
    return false;
  }

  const auto vrange_minmax = beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), 
    vrange.cend());
  const FLOAT_TYPE vrange_min = vrange_minmax.first;
  const FLOAT_TYPE vrange_max = vrange_minmax.second;
  const helperOC::DynSys_UMode_Type modified_uMode = 
    (schemeData->uMode == helperOC::DynSys_UMode_Default) ? 
    helperOC::DynSys_UMode_Max : schemeData->uMode;

  const helperOC::DynSys_DMode_Type modified_dMode = 
    (schemeData->dMode == helperOC::DynSys_DMode_Default) ? 
    helperOC::DynSys_DMode_Min : schemeData->dMode;

  return Car5D_CUDA::HamFunction_cuda(hamValue_uvec, x_uvecs, deriv_uvecs, 
    wMax, vrange_min, vrange_max, dMax, modified_uMode, modified_dMode, negate);
}


bool Car5D::PartialFunction_cuda(
    beacls::UVec& alpha_uvec,
    const DynSysSchemeData* schemeData,
    const FLOAT_TYPE,
    const beacls::UVec&,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& derivMin_uvecs,
    const std::vector<beacls::UVec>& derivMax_uvecs,
    const size_t dim,
    const size_t,
    const size_t) const {
  if (x_uvecs.size() < 3 || x_uvecs[2].empty() || derivMax_uvecs.size() < 3 || 
      derivMax_uvecs[0].empty() || derivMax_uvecs[1].empty() || 
      derivMax_uvecs[2].empty()) {
    return false; 
  }

  if (derivMax_uvecs.size() < 3 || derivMax_uvecs[0].empty() || 
    derivMax_uvecs[1].empty() || derivMax_uvecs[2].empty()) {
    return false;
  }

  const auto vrange_minmax = 
    beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());

  const FLOAT_TYPE vrange_min = vrange_minmax.first;
  const FLOAT_TYPE vrange_max = vrange_minmax.second;

  const helperOC::DynSys_UMode_Type modified_uMode = 
    (schemeData->uMode == helperOC::DynSys_UMode_Default) ? 
    helperOC::DynSys_UMode_Max : schemeData->uMode;

  const helperOC::DynSys_DMode_Type modified_dMode = 
    (schemeData->dMode == helperOC::DynSys_DMode_Default) ? 
    helperOC::DynSys_DMode_Min : schemeData->dMode;

  return Car5D_CUDA::PartialFunction_cuda(alpha_uvec, x_uvecs, derivMin_uvecs, 
    derivMax_uvecs, dim, wMax, vrange_min, vrange_max, dMax, modified_uMode, 
    modified_dMode);
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
