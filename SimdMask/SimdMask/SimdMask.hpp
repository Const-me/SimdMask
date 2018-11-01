#pragma once
#include <stdint.h>
#include <assert.h>
#include <utility>
#include <limits>
#include "SimdMaskTraits.hpp"

// This template class packs results of multiple vector tests in a single scalar.
// The default template parameters are for SSE1, they use 32-bit scalar, that's enough for 8 conditions. For wider SIMD or more conditions, you can switch to 64 bit scalars.
template<class eKey, class tMask = uint32_t, class tReg = __m128, class tTraits = SimdMaskTraits<tReg>>
class SimdMask
{
	tMask m_mask;

	// Count of lanes per register.
	static constexpr int nLanes = tTraits::nLanes;

	// Contains "1" for all lanes in test #0
	static constexpr tMask allSetMask = ( ~( (tMask)0 ) ) >> ( sizeof( tMask ) * 8 - nLanes );

	template<eKey v>
	static constexpr __forceinline int shiftBits()
	{
		static_assert( std::is_integral<tMask>::value, "Mask must be integer" );
		static_assert( !std::numeric_limits<tMask>::is_signed, "Mask must be unsigned" );

		constexpr int i = (int)v;
		static_assert( ( i + 1 ) * nLanes <= sizeof( tMask ) * 8, "The key is too large." );
		return i * nLanes;
	}

	template<eKey v>
	__forceinline void combine_or( int mask )
	{
		m_mask |= ( (tMask)( mask ) << shiftBits<v>() );
	}

	template<eKey v>
	static constexpr __forceinline tMask valueMask()
	{
		return ( (tMask)allSetMask ) << shiftBits<v>();
	}

public:
	// Set all bits to 0
	__forceinline SimdMask() : m_mask( (tMask)0 ) { }
	// Copy from another one.
	__forceinline SimdMask( const SimdMask& that ) : m_mask( that.m_mask ) { }

	// Reset everything
	__forceinline void clearEverything()
	{
		m_mask = (tMask)0;
	}

	// Clear all lanes for the specified condition.
	template<eKey v>
	__forceinline void clear()
	{
		constexpr tMask bits = ( allSetMask << shiftBits<v>() );
		m_mask &= ( ~bits );
	}

	// Merge, using bitwise |, with the values from the vector.
	template<eKey v>
	__forceinline void setVector( tReg r )
	{
		const int mask = tTraits::fromVector( r );
		combine_or<v>( mask );
	}

	// Merge, using bitwise |, with all true values for a specific test.
	template<eKey v>
	__forceinline void setAll()
	{
		combine_or<v>( (int)allSetMask );
	}

	// Check if every lane for the condition tested negatively.
	template<eKey v>
	__forceinline bool False() const
	{
		return 0 == ( m_mask & valueMask<v>() );
	}

	// Check if every lane for the condition tested positively.
	template<eKey v>
	__forceinline bool True() const
	{
		constexpr tMask m = valueMask<v>();
		return m == ( m_mask & m );
	}

	// Check specific lane for specific condition.
	template<eKey v, int lane>
	__forceinline bool condition() const
	{
		static_assert( lane >= 0 && lane < nLanes, "Invalid lane index" );
		constexpr tMask m = ( (tMask)( 1 ) ) << ( shiftBits<v>() + lane );
		return 0 != ( m_mask & m );
	}

	// Check specific lane for specific condition. Unlike the previous one, the lane here can be non-const, however the performance is slightly better when they're constants.
	template<eKey v>
	__forceinline bool condition( int lane ) const
	{
		assert( lane >= 0 && lane < nLanes );
		const tMask m = ( (tMask)( 1 ) ) << ( shiftBits<v>() + lane );
		return 0 != ( m_mask & m );
	}
};