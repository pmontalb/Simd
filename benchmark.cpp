
#include "Profiler/Profiler.h"
#include "CommandLineParser.h"
#include "Simd.h"

struct Config: public perf::Config
{
};
class Double2Profiler: public perf::Profiler<Double2Profiler, Config>
{
	friend class perf::Profiler<Double2Profiler, Config>;
public:
	using perf::Profiler<Double2Profiler, Config>::Profiler;
	std::string GetId() const noexcept { return "SIMD D2"; }
private:
	inline void OnStartImpl() noexcept { }

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		simd::Double2 x(x1, x2);
		simd::Double2 y = x.SquareRoot();
		y += one;
		y.Get(x1, x2);
	}

public:
	double x1 = 2.0;
	double x2 = 3.0;

	const simd::Double2 one { 1.0, 1.0 };
};

class TwoDoubleProfiler: public perf::Profiler<TwoDoubleProfiler, Config>
{
	friend class perf::Profiler<TwoDoubleProfiler, Config>;
public:
	using perf::Profiler<TwoDoubleProfiler, Config>::Profiler;
	std::string GetId() const noexcept { return "Scalar D2"; }
private:
	inline void OnStartImpl() noexcept { }

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		x1 = std::sqrt(x1) + 1.0;
		x2 = std::sqrt(x2) + 1.0;
	}

public:
	double x1 = 2.0;
	double x2 = 3.0;
};

class Float4Profiler: public perf::Profiler<Float4Profiler, Config>
{
	friend class perf::Profiler<Float4Profiler, Config>;
public:
	using perf::Profiler<Float4Profiler, Config>::Profiler;
	std::string GetId() const noexcept { return "SIMD F4"; }
private:
	inline void OnStartImpl() noexcept { }

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		simd::Float4 x(x1, x2, x3, x4);
		simd::Float4 y = x.SquareRoot();
		y += one;
		y.Get(x1, x2, x3, x4);
	}

public:
	float x1 = 2.0f;
	float x2 = 3.0f;
	float x3 = 5.0f;
	float x4 = 6.0f;

	const simd::Float4 one { 1.0, 1.0, 1.0, 1.0 };
};

class FourFloatProfiler: public perf::Profiler<FourFloatProfiler, Config>
{
	friend class perf::Profiler<FourFloatProfiler, Config>;
public:
	using perf::Profiler<FourFloatProfiler, Config>::Profiler;
	std::string GetId() const noexcept { return "Scalar F4"; }
private:
	inline void OnStartImpl() noexcept { }

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		x1 = std::sqrt(x1) + 1.0f;
		x2 = std::sqrt(x2) + 1.0f;
		x3 = std::sqrt(x3) + 1.0f;
		x4 = std::sqrt(x4) + 1.0f;
	}

public:
	float x1 = 2.0f;
	float x2 = 3.0f;
	float x3 = 5.0f;
	float x4 = 6.0f;

	const simd::Float4 one { 1.0, 1.0, 1.0, 1.0 };
};

class GroupingDouble2Profiler: public perf::Profiler<GroupingDouble2Profiler, Config>
{
	friend class perf::Profiler<GroupingDouble2Profiler, Config>;
public:
	using perf::Profiler<GroupingDouble2Profiler, Config>::Profiler;
	std::string GetId() const noexcept { return "SIMD GROUP D2"; }
private:
	inline void OnStartImpl() noexcept { }

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		auto x = simd::Double2(x1, x2);
		x += simd::Double2(x3, x4);
		x.Get(x1, x2);

		x = simd::sqrt(x);
		x.Get(x3, x4);
	}

public:
	double x1 = 2.0;
	double x2 = 3.0;

	double x3 = 4.0;
	double x4 = 5.0;
};

class TwoDoubleGroupingProfiler: public perf::Profiler<TwoDoubleGroupingProfiler, Config>
{
	friend class perf::Profiler<TwoDoubleGroupingProfiler, Config>;
public:
	using perf::Profiler<TwoDoubleGroupingProfiler, Config>::Profiler;
	std::string GetId() const noexcept { return "Scalar GROUP D2"; }
private:
	inline void OnStartImpl() noexcept { }

	inline void OnEndImpl() noexcept { }

	inline void RunImpl() noexcept
	{
		x1 += x3;
		x2 += x4;

		x3 = std::sqrt(x1);
		x4 = std::sqrt(x2);
	}

public:
	double x1 = 2.0;
	double x2 = 3.0;

	double x3 = 4.0;
	double x4 = 5.0;
};


template<typename Profiler1T, typename Profiler2T>
void Run(const clp::CommandLineArgumentParser& ap)
{
	Config config;
	config.nIterations = ap.GetArgumentValue<size_t>("-nit");
	config.nIterationsPerCycle = ap.GetArgumentValue<size_t>("-npc");
	config.nWarmUpIterations = ap.GetArgumentValue<size_t>("-wui");
	config.timeScale = perf::TimeScale::Nanoseconds;

	Profiler1T profilerSimd(config);
	if (ap.GetFlag("-cg"))
		profilerSimd.Instrument();
	else
		profilerSimd.Profile();

	if (ap.GetFlag("-v"))
	{
		std::cout << "*****" << profilerSimd.GetId() << "*****" << std::endl;
		profilerSimd.Print();
		std::cout << "x1=" << profilerSimd.x1 << "|x2=" << profilerSimd.x2 << std::endl;
		std::cout << "**********" << std::endl;
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
		std::cout << "*****" << profilerScalar.GetId() << "*****" << std::endl;
		profilerScalar.Print();
		std::cout << "x1=" << profilerScalar.x1 << "|x2=" << profilerScalar.x2 << std::endl;
		std::cout << "**********" << std::endl;
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
	else if (ap.GetArgumentValue<std::string>("-mode") == "GD2")
	{
		if (ap.GetFlag("-reverse"))
			Run<TwoDoubleGroupingProfiler, GroupingDouble2Profiler>(ap);
		else
			Run<GroupingDouble2Profiler, TwoDoubleGroupingProfiler>(ap);
	}
}
