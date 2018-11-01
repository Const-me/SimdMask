#pragma once
#include <xmmintrin.h>

// The mask values must be either exactly 0, or all ones.
__forceinline __m128 blendv_ps( __m128 a, __m128 b, __m128 maskPickB )
{
	using namespace Intrinsics::Sse;
	b = and_ps( b, maskPickB );
	a = andnot_ps( maskPickB, a );
	return or_ps( a, b );
}

__m128 __vectorcall randomValue( double i, double ax );