#pragma once
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

namespace
{
	template<class tReg>
	struct SimdMaskTraits;

	// SSE1, single precision floats
	template<>
	struct SimdMaskTraits<__m128>
	{
		static constexpr int nLanes = 4;

		static int __forceinline fromVector( __m128 v )
		{
			return _mm_movemask_ps( v );
		}
	};

	// SSE2, double precision floats
	template<>
	struct SimdMaskTraits<__m128d>
	{
		static constexpr int nLanes = 2;

		static int __forceinline fromVector( __m128d v )
		{
			return _mm_movemask_pd( v );
		}
	};

	// AVX1, single precision floats
	template<>
	struct SimdMaskTraits<__m256>
	{
		static constexpr int nLanes = 8;

		static int __forceinline fromVector( __m256 v )
		{
			return _mm256_movemask_ps( v );
		}
	};

	// AVX1, double precision floats
	template<>
	struct SimdMaskTraits<__m256d>
	{
		static constexpr int nLanes = 4;

		static int __forceinline fromVector( __m256d v )
		{
			return _mm256_movemask_pd( v );
		}
	};
}