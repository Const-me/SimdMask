#include "stdafx.h"
#include "miscUtils.h"
#include <random>
#include <array>

static uint32_t rdrand()
{
	uint32_t res;
	_rdrand32_step( &res );
	return res;
}

__m128 __vectorcall randomValue( double i, double ax )
{
	static std::default_random_engine generator{ rdrand() };
	std::uniform_real_distribution<float> distribution;
	alignas( 16 ) std::array<float, 4> result;
	for( float& x : result )
		x = distribution( generator );

	using namespace Intrinsics::Sse;

	__m128 r = load_ps( result.data() );
	r *= set1_ps( (float)( ax - i ) );
	r += set1_ps( (float)i );
	return r;
}