
#include "Profiler/Profiler.h"
#include "CommandLineParser.h"
#include "Simd.h"

struct Config: public perf::Config
{
	size_t vectorSize = 256;
};
class Double2Profiler: public perf::Profiler<Double2Profiler, Config>
{
	friend class perf::Profiler<Double2Profiler, Config>;
public:
	using perf::Profiler<Double2Profiler, Config>::Profiler;
	std::string GetId() const noexcept { return "SIMD D2"; }
private:
	inline void OnStartImpl() noexcept
	{
		vector.resize(_config.vectorSize, 1.0);
	}

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		for (size_t i = 0; i < (vector.size() + 1) / 2; ++i)
		{
			if (2 * i <= vector.size() - 2)
			{
				simd::Double2 x(vector.data() + 2 * i);
				x += lambda;
				x = x.SquareRoot();
				x.Get(vector.data() + 2 * i);
			}
			else
			{
				vector[2 * i] = std::sqrt(vector[2 * i] + 1.0);
			}
		}
	}

public:
	simd::Double2::AlignedVector vector {};
	const simd::Double2 lambda { 1.0, 1.0 };
};

class TwoDoubleProfiler: public perf::Profiler<TwoDoubleProfiler, Config>
{
	friend class perf::Profiler<TwoDoubleProfiler, Config>;
public:
	using perf::Profiler<TwoDoubleProfiler, Config>::Profiler;
	std::string GetId() const noexcept { return "Scalar D2"; }
private:
	inline void OnStartImpl() noexcept
	{
		vector.resize(_config.vectorSize, 1.0);
	}

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		for (size_t i = 0; i < vector.size(); ++i)
			vector[i] = std::sqrt(vector[i] + 1.0);
	}

public:
	simd::Double2::AlignedVector vector {};
};

class Float4Profiler: public perf::Profiler<Float4Profiler, Config>
{
	friend class perf::Profiler<Float4Profiler, Config>;
public:
	using perf::Profiler<Float4Profiler, Config>::Profiler;
	std::string GetId() const noexcept { return "SIMD F4"; }
private:
	inline void OnStartImpl() noexcept
	{
		vector.resize(_config.vectorSize, 1.0);
	}

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		for (size_t i = 0; i < (vector.size() + 3) / 4; ++i)
		{
			if (4 * i <= vector.size() - 4)
			{
				simd::Float4 x(vector.data() + 4 * i);
				x += lambda;
				x = x.SquareRoot();
				x.Get(vector.data() + 4 * i);
			}
			else
			{
				for (size_t j = 4 * i; j < vector.size(); ++j)
					vector[j] = std::sqrt(vector[j] + 1.0f);
			}
		}
	}

public:
	simd::Float4::AlignedVector vector {};
	const simd::Float4 lambda { 1.0, 1.0, 1.0, 1.0 };
};

class FourFloatProfiler: public perf::Profiler<FourFloatProfiler, Config>
{
	friend class perf::Profiler<FourFloatProfiler, Config>;
public:
	using perf::Profiler<FourFloatProfiler, Config>::Profiler;
	std::string GetId() const noexcept { return "Scalar F4"; }
private:
	inline void OnStartImpl() noexcept
	{
		vector.resize(_config.vectorSize, 1.0);
	}

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		for (size_t i = 0; i < vector.size(); ++i)
			vector[i] = std::sqrt(vector[i] + 1.0f);
	}

public:
	simd::Float4::AlignedVector vector {};
};

#ifdef __AVX__
	class Float8Profiler: public perf::Profiler<Float8Profiler, Config>
	{
		friend class perf::Profiler<Float8Profiler, Config>;
	public:
		using perf::Profiler<Float8Profiler, Config>::Profiler;
		std::string GetId() const noexcept { return "SIMD F8"; }
	private:
		inline void OnStartImpl() noexcept
		{
			vector.resize(_config.vectorSize, 1.0);

			simd::Float8::AlignedArray tmp;
			tmp.fill(1.0);
			lambda = simd::Float8(std::move(tmp));
		}

		inline void OnEndImpl() noexcept { }

		inline void RunImpl() noexcept
		{
			for (size_t i = 0; i < (vector.size() + 7) / 8; ++i)
			{
				if (8 * i <= vector.size() - 8)
				{
					simd::Float8 x(vector.data() + 8 * i);
					x += lambda;
					x = x.SquareRoot();
					x.Get(vector.data() + 8 * i);
				}
				else
				{
					for (size_t j = 8 * i; j < vector.size(); ++j)
						vector[j] = std::sqrt(vector[j] + 1.0f);
				}
			}
		}

	public:
		simd::Float8::AlignedVector vector {};
		simd::Float8 lambda {};
	};

class EightFloatProfiler: public perf::Profiler<EightFloatProfiler, Config>
{
	friend class perf::Profiler<EightFloatProfiler, Config>;
public:
	using perf::Profiler<EightFloatProfiler, Config>::Profiler;
	std::string GetId() const noexcept { return "Scalar F8"; }
private:
	inline void OnStartImpl() noexcept
	{
		vector.resize(_config.vectorSize, 1.0);
	}

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		for (size_t i = 0; i < vector.size(); ++i)
			vector[i] = std::sqrt(vector[i] + 1.0f);
	}

public:
	simd::Float8::AlignedVector vector {};
};

#endif

template<typename Profiler1T, typename Profiler2T>
void Run(const clp::CommandLineArgumentParser& ap)
{
	Config config;
	config.nIterations = ap.GetArgumentValue<size_t>("-nit");
	config.nIterationsPerCycle = ap.GetArgumentValue<size_t>("-npc");
	config.nWarmUpIterations = ap.GetArgumentValue<size_t>("-wui");
	config.timeScale = perf::TimeScale::Nanoseconds;

	config.vectorSize = ap.GetArgumentValue<size_t>("-sz", 257);

	Profiler1T profilerSimd(config);
	if (ap.GetFlag("-cg"))
		profilerSimd.Instrument();
	else
		profilerSimd.Profile();

	if (ap.GetFlag("-v"))
	{
		profilerSimd.Print();
		std::cout << "simd::x1=" << profilerSimd.vector[0] << std::endl;
	}
	else if (ap.GetFlag("-py"))
	{
		profilerSimd.PrintPythonPlotInstructions(false, profilerSimd.GetId());
	}

	Profiler2T profilerScalar(config);
	if (ap.GetFlag("-cg"))
		profilerScalar.Instrument();
	else
		profilerScalar.Profile();

	if (ap.GetFlag("-v"))
	{
		profilerScalar.Print();
		std::cout << "scalar::x1=" << profilerScalar.vector[0] << std::endl;
	}
	else if (ap.GetFlag("-py"))
	{
		profilerScalar.PrintPythonPlotInstructions(true, profilerScalar.GetId());
	}
}

int main(int argc, char** argv)
{
	clp::CommandLineArgumentParser ap(argc, argv);

	if (ap.GetArgumentValue<std::string>("-mode") == "D2")
	{
		if (ap.GetFlag("-reverse"))
			Run<TwoDoubleProfiler, Double2Profiler>(ap);
		else
			Run<Double2Profiler, TwoDoubleProfiler>(ap);
	}
	else if (ap.GetArgumentValue<std::string>("-mode") == "F4")
	{
		if (ap.GetFlag("-reverse"))
			Run<FourFloatProfiler, Float4Profiler>(ap);
		else
			Run<Float4Profiler, FourFloatProfiler>(ap);
	}
	#ifdef __AVX__
		else if (ap.GetArgumentValue<std::string>("-mode") == "F8")
		{
			if (ap.GetFlag("-reverse"))
				Run<EightFloatProfiler, Float8Profiler>(ap);
			else
				Run<Float8Profiler, EightFloatProfiler>(ap);
		}
		else if (ap.GetArgumentValue<std::string>("-mode") == "F8F4")
		{
			if (ap.GetFlag("-reverse"))
				Run<Float4Profiler, Float8Profiler>(ap);
			else
				Run<Float8Profiler, Float4Profiler>(ap);
		}
	#endif
}
