#include "stdafx.h"
#include "SimdMask/SimdMask.hpp"
#include "miscUtils.h"

using namespace Intrinsics::Sse;

namespace
{
	// Various vector tests we need for solving these equations
	enum struct eTest : uint8_t
	{
		// a != 0.0, i.e. the equation is actually quadratic
		Quadratic,
		// The discriminant is positive
		DiscrPositive,
		// The discriminant is negative
		DiscrNegative,
		// b = 0.0
		ZeroB,
	};

	// Specialize the generic SimdMask class for this particular problem.
	class Mask : public SimdMask<eTest>
	{
		template<class tFunc>
		__forceinline __m128i countRoots( tFunc fn ) const
		{
			// BTW, the specs say taking address of inline function may prevent inlining. Fortunately for us, VC++ still inlines just fine.
			return set_epi32( (this->*fn)( 0 ), ( this->*fn )( 1 ), ( this->*fn )( 2 ), ( this->*fn )( 3 ) );
		}

		__forceinline int quadraticRoots( int lane ) const
		{
			if( condition<eTest::DiscrNegative>( lane ) )
				return 0;
			if( condition<eTest::DiscrPositive>( lane ) )
				return 2;
			return 1;
		}

		__forceinline int linearRoots( int lane ) const
		{
			if( !condition<eTest::ZeroB>( lane ) )
				return 1;
			return 0;
		}

		__forceinline int generalRoots( int lane ) const
		{
			if( condition<eTest::Quadratic>( lane ) )
				return quadraticRoots( lane );
			return linearRoots( lane );
		}

	public:
		// Calculate roots count when all lanes contain quadratic equations
		__forceinline __m128i countQuadraticRoots() const
		{
			return countRoots( &Mask::quadraticRoots );
		}
		// Calculate roots count when all lanes contain linear equations
		__forceinline __m128i countLinearRoots() const
		{
			return countRoots( &Mask::linearRoots );
		}
		// Calculate roots count for general case, this one is the slowest of all 3.
		__forceinline __m128i countAllRoots() const
		{
			return countRoots( &Mask::generalRoots );
		}
	};

	// The solutions of these equations
	struct Solution
	{
		// Count of roots in 32-bit integer lanes, between 0 and 2.
		__m128i rootsCount;

		// The roots of the equations. For lanes where count = 1, corresponding r2 values are garbage. For lanes where count = 0, both r1 and r2 are garbage.
		__m128 r1, r2;

		// Calculate ( b ± sqrt( discr ) ) / ( 2 * a )
		__forceinline void quadraticFormula( __m128 a, __m128 b, __m128 discr )
		{
			const __m128 discrSqrt = sqrt_ps( discr );
			const __m128 mul = set1_ps( 0.5f ) / a;
			r1 = ( b - discrSqrt ) * mul;
			r2 = ( b + discrSqrt ) * mul;
		}

		__forceinline void sameRootsCount( int c )
		{
			assert( c >= 0 && c <= 2 );
			rootsCount = set1_epi32( c );
		}
	};
}

// noinline is here just for convenience of using the disassembler. In production code you'll want to inline as much as possible.
__declspec( noinline ) Solution __vectorcall solveQuadratic( __m128 a, __m128 b, __m128 c )
{
	const __m128 signBits = set1_ps( -0.0f );

	// Flip signs so that a >= 0: this will ensure the returned roots are sorted i.e. r1 < r2
	{
		const __m128 aSign = a & signBits;
		a ^= aSign;
		b ^= aSign;
		c ^= aSign;
	}
	// b = -b: both quadratic and linear formulas need a negative of b.
	b ^= signBits;

	Mask mask;
	const __m128 zero = setzero_ps();
	Solution result;

	// First couple of tests
	const __m128 quadraticMask = cmpneq_ps( a, zero );
	mask.setVector<eTest::Quadratic>( quadraticMask );

	const __m128 discr = b * b + set1_ps( -4.0f ) * a * c;
	mask.setVector<eTest::DiscrPositive>( cmpgt_ps( discr, zero ) );

	// Check if all lanes have quadratic equations with 2 roots. If that's the case, we can compute the solutions right away, without any more scalar code or conditions.
	if( mask.True<eTest::Quadratic>() && mask.True<eTest::DiscrPositive>() )
	{
		// BTW, the above "if" compiled to a single instruction, this one: cmp al,0FFh. Very efficient.
		result.quadraticFormula( a, b, discr );
		result.sameRootsCount( 2 );
		return result;
	}

	mask.setVector<eTest::DiscrNegative>( cmple_ps( discr, zero ) );

	if( mask.True<eTest::Quadratic>() )
	{
		// All lanes have quadratic equations in them..
		if( mask.True<eTest::DiscrNegative>() )
		{
			// .. all with no roots.
			result.rootsCount = setzero_all();
			return result;
		}

		// We already handled True(Quadratic) && True(DiscrPositive) above that.
		// If we're here, it means different count of roots in different lanes. Or all have 1 root, if you expect your data will have this, add another fully-vectorized version when False<DiscrNegative> && False<DiscrPositive>
		result.quadraticFormula( a, b, discr );
		result.rootsCount = mask.countQuadraticRoots();
		return result;
	}

	mask.setVector<eTest::ZeroB>( cmpeq_ps( b, zero ) );

	if( mask.False<eTest::Quadratic>() )
	{
		// All lanes have linear equations in them.
		result.r1 = c / b;

		if( mask.False<eTest::ZeroB>() )
			result.sameRootsCount( 1 );
		else
			result.rootsCount = mask.countLinearRoots();
		return result;
	}

	// All lanes are different. Calculate roots using per-lane blending, and count roots with the most generic method. This is the worst case performance-wise.
	result.quadraticFormula( a, b, discr );
	result.r1 = blendv_ps( c / b, result.r1, quadraticMask );
	result.rootsCount = mask.countAllRoots();
	return result;
}

// Scalar version
__forceinline int solveQuadratic_scalar( float a, float b, float c, float& r1, float& r2 )
{
	if( a < 0 )
	{
		a = -a;
		c = -c;
	}
	else
		b = -b;

	if( a != 0.0f )
	{
		// Quadratic
		const float d = b * b - 4.0f * a * c;
		if( d < 0.0f )
			return 0;
		const float mul = 0.5f / a;
		if( d > 0.0f )
		{
			const float sq = sqrtf( d );
			r1 = ( b - sq ) * mul;
			r2 = ( b + sq ) * mul;
			return 2;
		}
		r1 = b * mul;
		return 1;
	}

	// Linear
	if( 0.0f != b )
	{
		r1 = c / b;
		return 1;
	}
	return 0;
}

static void printRoots( int c, float r1, float r2 )
{
	switch( c )
	{
	case 0:
		printf( "no roots" ); return;
	case 1:
		printf( "1 root: %f", r1 ); return;
	case 2:
		printf( "2 roots: %f, %f", r1, r2 ); return;
	}
	assert( false );
}

void test1()
{
	const __m128 a = randomValue( 0.8, 1.2 );
	const __m128 b = randomValue( -0.25, 0.25 );
	// const __m128 c = randomValue( -5, -3 );
	const __m128 c = randomValue( -4, 1 );
	const Solution result = solveQuadratic( a, b, c );
	for( int i = 0; i < 4; i++ )
	{
		const float _a = a.m128_f32[ i ];
		const float _b = b.m128_f32[ i ];
		const float _c = c.m128_f32[ i ];
		const int nRoots = result.rootsCount.m128i_i32[ i ];
		printf( "%f, %f, %f -> ", _a, _b, _c );
		printRoots( nRoots, result.r1.m128_f32[ i ], result.r2.m128_f32[ i ] );
		printf( "\n\tscalar solver -> " );

		float r1, r2;
		const int rootsScalar = solveQuadratic_scalar( _a, _b, _c, r1, r2 );
		printRoots( rootsScalar, r1, r2 );
		printf( "\n" );
	}
}

int main()
{
	test1();
}