#ifndef __Car5D_hpp__
#define __Car5D_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <typedef.hpp>
#include <cstddef>
#include <vector>
#include <iostream>
#include <cstring>
#include <utility>
using namespace std::rel_ops;

namespace helperOC {
	/*
		@note: Since plane is a "handle class", we can pass on
		handles/pointers to other plane objects
		e.g. a.platoon.leader = b (passes b by reference, does not create a copy)
		Also see constructor
	*/
	class Car5D : public DynSys {
	public:
	protected:
		beacls::FloatVec uRange;	// Tracking control range
		beacls::FloatVec aRange;	// Planning control range (acceleration)
		beacls::FloatVec dRange;	// Disturbance range (acceleration)

		const FLOAT_TYPE n0 = 10;
		const FLOAT_TYPE d1 = 8;
		const FLOAT_TYPE d0 = 10;
		const FLOAT_TYPE g = 9.81;
		const FLOAT_TYPE m = 1.3;
	public:
		/*
      Dynamics:
          \dot x_1 = x_2                      + d{1}
          \dot x_2 = g*tan(x_3)               - d{2}
          \dot x_3 = -d1*x_3 + x_4
          \dot x_4 = -d0*x_3       + n0*u{1}
              uMin <= [u1; u2] <= uMax
		@param	[in]		x	state [xpos; ypos; theta]
		@param	[in]		wMax	maximum turn rate
		@param	[in]		vrange	speed range
		@param	[in]		dMax	disturbance bounds
		@return	a Plane object
		*/
		PREFIX_VC_DLL
			Car5D(
				const beacls::FloatVec& x,
				const beacls::FloatVec& uRange,
				const beacls::FloatVec& arange,
				const beacls::FloatVec& dRange);

		PREFIX_VC_DLL
			virtual ~Car5D();

		PREFIX_VC_DLL
			virtual bool operator==(const Car5D& rhs) const;

		PREFIX_VC_DLL
			virtual bool operator==(const DynSys& rhs) const;

		virtual Car5D* clone() const {
			return new Car5D(*this);
		}
		/*
		@brief Control of Dubins Car
		*/
		PREFIX_VC_DLL
			bool optCtrl(
				std::vector<beacls::FloatVec >& uOpts,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& y_ites,
				const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
				const beacls::IntegerVec& y_sizes,
				const beacls::IntegerVec& deriv_sizes,
				const helperOC::DynSys_UMode_Type uMode
			) const;

		PREFIX_VC_DLL
			bool optDstb(
				std::vector<beacls::FloatVec >& uOpts,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& y_ites,
				const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
				const beacls::IntegerVec& y_sizes,
				const beacls::IntegerVec& deriv_sizes,
				const helperOC::DynSys_DMode_Type dMode
			) const;

		PREFIX_VC_DLL
			bool dynamics(
				std::vector<beacls::FloatVec >& dx,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& x_ites,
				const std::vector<beacls::FloatVec >& us,
				const std::vector<beacls::FloatVec >& ds,
				const beacls::IntegerVec& x_sizes,
				const size_t dst_target_dim
			) const;

// ============================================ GPU STUFF BELOW
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
		/*
		@brief Control of Dubins Car
		*/
		PREFIX_VC_DLL
			bool optCtrl_cuda(
				std::vector<beacls::UVec>& u_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_uvecs,
				const helperOC::DynSys_UMode_Type uMode
			) const;
		/*
		@brief disturbance of Dubins Car
		\dot{x}_1 = v * cos(x_3) + d_1
		\dot{x}_2 = v * sin(x_3) + d_2
		\dot{x}_3 = u
		*/
		PREFIX_VC_DLL
			bool optDstb_cuda(
				std::vector<beacls::UVec>& d_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_uvecs,
				const helperOC::DynSys_DMode_Type dMode
			) const;
		/*
		@brief Dynamics of the Dubins Car
		\dot{x}_1 = v * cos(x_3)
		\dot{x}_2 = v * sin(x_3)
		\dot{x}_3 = w
		Control: u = w;
		@author Mo Chen, 2016-06-08
		*/
		PREFIX_VC_DLL
			bool dynamics_cuda(
				std::vector<beacls::UVec>& dx_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& u_uvecs,
				const std::vector<beacls::UVec>& d_uvecs,
				const size_t dst_target_dim
			) const;
		PREFIX_VC_DLL
			bool optCtrl_cuda(
				std::vector<beacls::UVec>& uL_uvecs,
				std::vector<beacls::UVec>& uU_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMin_uvecs,
				const std::vector<beacls::UVec>& derivMax_uvecs,
				const helperOC::DynSys_UMode_Type uMode = helperOC::DynSys_UMode_Default
			) const;
		PREFIX_VC_DLL
			bool optDstb_cuda(
				std::vector<beacls::UVec>& dL_uvecs,
				std::vector<beacls::UVec>& dU_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMin_uvecs,
				const std::vector<beacls::UVec>& derivMax_uvecs,
				const helperOC::DynSys_DMode_Type dMode = helperOC::DynSys_DMode_Default
			) const;
		PREFIX_VC_DLL
			bool dynamics_cuda(
				beacls::UVec& alpha_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& uL_uvecs,
				const std::vector<beacls::UVec>& uU_uvecs,
				const std::vector<beacls::UVec>& dL_uvecs,
				const std::vector<beacls::UVec>& dU_uvecs,
				const size_t dst_target_dim = std::numeric_limits<size_t>::max()
			) const;
		PREFIX_VC_DLL
			bool HamFunction_cuda(
				beacls::UVec& hamValue_uvec,
				const DynSysSchemeData* schemeData,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivs,
				const size_t begin_index,
				const size_t length,
				const bool negate
			) const;
		PREFIX_VC_DLL
			bool PartialFunction_cuda(
				beacls::UVec& alpha_uvec,
				const DynSysSchemeData*,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMin_uvecs,
				const std::vector<beacls::UVec>& derivMax_uvecs,
				const size_t dim,
				const size_t begin_index,
				const size_t length
			) const;
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */

	protected:
		/** @overload
		Disable copy constructor
		*/
		PREFIX_VC_DLL
			Car5D(const Car5D& rhs) :
			DynSys(rhs),
			uRange(rhs.uRange),	//!< Angular control bounds
			aRange(rhs.aRange),	//!< Speed control bounds
			dRange(rhs.dRange)	//!< Disturbance
		{}
	private:
		/** @overload
		Disable operator=
		*/
		Car5D& operator=(const Car5D& rhs);
		bool dynamics_cell_helper(beacls::FloatVec& dx,
				const beacls::FloatVec::const_iterator& x_ite1,
				const beacls::FloatVec::const_iterator& x_ite2,
				const beacls::FloatVec::const_iterator& x_ite3,
				const std::vector<beacls::FloatVec >& us,
				const std::vector<beacls::FloatVec >& ds,
				const size_t x_size,
				const size_t dim) const;
	};
};
#endif	/* __Car5D_hpp__ */
