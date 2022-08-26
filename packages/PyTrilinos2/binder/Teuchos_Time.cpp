#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_Comm.hpp> // Teuchos::Comm
#include <Teuchos_CommandLineProcessor.hpp> // Teuchos::CommandLineProcessor
#include <Teuchos_CommandLineProcessor.hpp> // Teuchos::CommandLineProcessor::HelpPrinted
#include <Teuchos_CommandLineProcessor.hpp> // Teuchos::CommandLineProcessor::ParseError
#include <Teuchos_CommandLineProcessor.hpp> // Teuchos::CommandLineProcessor::TimeMonitorSurrogate
#include <Teuchos_CommandLineProcessor.hpp> // Teuchos::CommandLineProcessor::UnrecognizedOption
#include <Teuchos_CompileTimeAssert.hpp> // Teuchos::CompileTimeAssert
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_OpaqueWrapper.hpp> // Teuchos::OpaqueWrapper
#include <Teuchos_ParameterList.hpp> // Teuchos::ParameterList
#include <Teuchos_PerformanceMonitorBase.hpp> // Teuchos::ECounterSetOp
#include <Teuchos_PtrDecl.hpp> // Teuchos::Ptr
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPNodeLookup
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_ReductionOp.hpp> // Teuchos::ValueTypeReductionOp
#include <Teuchos_ReductionOpHelpers.hpp> // Teuchos::CharToValueTypeReductionOp
#include <Teuchos_ReductionOpHelpers.hpp> // Teuchos::CharToValueTypeReductionOpImp
#include <Teuchos_SerializationTraits.hpp> // Teuchos::SerializationTraits
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ConstValueTypeDeserializationBuffer
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ConstValueTypeDeserializationBufferImp
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ConstValueTypeSerializationBuffer
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ConstValueTypeSerializationBufferImp
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::DefaultSerializer
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ValueTypeDeserializationBuffer
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ValueTypeDeserializationBufferImp
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ValueTypeSerializationBuffer
#include <Teuchos_SerializationTraitsHelpers.hpp> // Teuchos::ValueTypeSerializationBufferImp
#include <Teuchos_Serializer.hpp> // Teuchos::Serializer
#include <Teuchos_Time.hpp> // Teuchos::Time
#include <Teuchos_TimeMonitor.hpp> // Teuchos::SyncTimeMonitor
#include <Teuchos_TimeMonitor.hpp> // Teuchos::TimeMonitor
#include <Teuchos_TimeMonitor.hpp> // Teuchos::TimeMonitorSurrogateImpl
#include <Teuchos_TimeMonitor.hpp> // Teuchos::TimeMonitorSurrogateImplInserter
#include <Teuchos_Workspace.hpp> // Teuchos::RawWorkspace
#include <Teuchos_Workspace.hpp> // Teuchos::WorkspaceStore
#include <Teuchos_Workspace.hpp> // Teuchos::WorkspaceStoreInitializeable
#include <Teuchos_Workspace.hpp> // Teuchos::print_memory_usage_stats
#include <cwchar> // (anonymous)
#include <ios> // std::_Ios_Openmode
#include <ios> // std::_Ios_Seekdir
#include <ios> // std::fpos
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <mpi.h> // ompi_communicator_t
#include <mpi.h> // ompi_errhandler_t
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Teuchos_RCP.hpp>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, Teuchos::RCP<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// Teuchos::TimeMonitorSurrogateImpl file:Teuchos_TimeMonitor.hpp line:812
struct PyCallBack_Teuchos_TimeMonitorSurrogateImpl : public Teuchos::TimeMonitorSurrogateImpl {
	using Teuchos::TimeMonitorSurrogateImpl::TimeMonitorSurrogateImpl;

};

void bind_Teuchos_Time(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Teuchos::Time file:Teuchos_Time.hpp line:85
		pybind11::class_<Teuchos::Time, Teuchos::RCP<Teuchos::Time>> cl(M("Teuchos"), "Time", "Wall-clock timer.\n\n To time a section of code, place it in between calls to start()\n and stop().  It is better to access this class through the\n TimeMonitor class (which see) for exception safety and correct\n behavior in reentrant code.\n\n If Teuchos is configured with TPL_ENABLE_Valgrind=ON\n and Teuchos_TIME_MASSIF_SNAPSHOTS=ON Valgrind Massif\n snapshots are taken before and after Teuchos::Time invocation. The\n resulting memory profile can be plotted using\n core/utils/plotMassifMemoryUsage.py");
		cl.def( pybind11::init( [](const std::string & a0){ return new Teuchos::Time(a0); } ), "doc" , pybind11::arg("name"));
		cl.def( pybind11::init<const std::string &, bool>(), pybind11::arg("name"), pybind11::arg("start") );

		cl.def( pybind11::init( [](Teuchos::Time const &o){ return new Teuchos::Time(o); } ) );
		cl.def_static("wallTime", (double (*)()) &Teuchos::Time::wallTime, "Current wall-clock time in seconds.\n\n This is only useful for measuring time intervals.  The absolute\n value returned is measured relative to some arbitrary time in\n the past.\n\nC++: Teuchos::Time::wallTime() --> double");
		cl.def("start", [](Teuchos::Time &o) -> void { return o.start(); }, "");
		cl.def("start", (void (Teuchos::Time::*)(bool)) &Teuchos::Time::start, "Start the timer, if the timer is enabled (see disable()).\n\n \n [in] If true, reset the timer's total elapsed time\n   to zero before starting the timer.  By default, the timer\n   accumulates the total elapsed time for all start() ... stop()\n   sequences.\n\nC++: Teuchos::Time::start(bool) --> void", pybind11::arg("reset"));
		cl.def("stop", (double (Teuchos::Time::*)()) &Teuchos::Time::stop, "Stop the timer, if the timer is enabled (see disable()).\n\nC++: Teuchos::Time::stop() --> double");
		cl.def("disable", (void (Teuchos::Time::*)()) &Teuchos::Time::disable, "\"Disable\" this timer, so that it ignores calls to start() and stop().\n\nC++: Teuchos::Time::disable() --> void");
		cl.def("enable", (void (Teuchos::Time::*)()) &Teuchos::Time::enable, "\"Enable\" this timer, so that it (again) respects calls to start() and stop().\n\nC++: Teuchos::Time::enable() --> void");
		cl.def("isEnabled", (bool (Teuchos::Time::*)() const) &Teuchos::Time::isEnabled, "Whether the timer is enabled (see disable()).\n\nC++: Teuchos::Time::isEnabled() const --> bool");
		cl.def("totalElapsedTime", [](Teuchos::Time const &o) -> double { return o.totalElapsedTime(); }, "");
		cl.def("totalElapsedTime", (double (Teuchos::Time::*)(bool) const) &Teuchos::Time::totalElapsedTime, "The total time in seconds accumulated by this timer.\n\n \n [in] If true, return the current elapsed\n   time since the first call to start() when the timer was\n   enabled, whether or not the timer is running or enabled.  If\n   false, return the total elapsed time as of the last call to\n   stop() when the timer was enabled.\n\n \n If start() has never been called when the timer was\n   enabled, and if readCurrentTime is true, this method will\n   return wallTime(), regardless of the actual start time.\n\nC++: Teuchos::Time::totalElapsedTime(bool) const --> double", pybind11::arg("readCurrentTime"));
		cl.def("reset", (void (Teuchos::Time::*)()) &Teuchos::Time::reset, "Reset the cummulative time and call count.\n\nC++: Teuchos::Time::reset() --> void");
		cl.def("isRunning", (bool (Teuchos::Time::*)() const) &Teuchos::Time::isRunning, "Whether the timer is currently running.\n\n \"Currently running\" means either that start() has been called\n without an intervening stop() call, or that the timer was\n created already running and stop() has not since been called.\n\nC++: Teuchos::Time::isRunning() const --> bool");
		cl.def("name", (const std::string & (Teuchos::Time::*)() const) &Teuchos::Time::name, "The name of this timer.\n\nC++: Teuchos::Time::name() const --> const std::string &", pybind11::return_value_policy::automatic);
		cl.def("incrementNumCalls", (void (Teuchos::Time::*)()) &Teuchos::Time::incrementNumCalls, "Increment the number of times this timer has been called,\n   if the timer is enabled (see disable()).\n\nC++: Teuchos::Time::incrementNumCalls() --> void");
		cl.def("numCalls", (int (Teuchos::Time::*)() const) &Teuchos::Time::numCalls, "The number of times this timer has been called while enabled.\n\nC++: Teuchos::Time::numCalls() const --> int");
	}
	{ // Teuchos::TimeMonitor file:Teuchos_TimeMonitor.hpp line:178
		pybind11::class_<Teuchos::TimeMonitor, Teuchos::RCP<Teuchos::TimeMonitor>, Teuchos::PerformanceMonitorBase<Teuchos::Time>> cl(M("Teuchos"), "TimeMonitor", "Scope guard for Time, that can compute MPI collective timer\n   statistics.\n\n An instance of the TimeMonitor class wraps a nonconst reference to\n a Time timer object.  TimeMonitor's constructor starts the timer,\n and its destructor stops the timer.  This ensures scope safety of\n timers, so that no matter how a scope is exited (whether the\n normal way or when an exception is thrown), a timer started in the\n scope is stopped when the scope is left.\n\n TimeMonitor also has class methods that create or destroy timers\n and compute global timer statistics.  If you create a timer using\n getNewCounter() (or the deprecated getNewTimer()), it will add\n that timer to the set of timers for which to compute global\n statistics.  The summarize() and report() methods will print\n global statistics for these timers, like the minimum, mean, and\n maximum time over all processes in the communicator, for each\n timer.  These methods work correctly even if some processes have\n different timers than other processes.  You may also use\n computeGlobalTimerStatistics() to compute the same global\n statistics, if you wish to use them in your program or output them\n in a different format than that of these methods.\n\n If Teuchos is configured with TPL_ENABLE_Valgrind=ON\n and Teuchos_TIME_MASSIF_SNAPSHOTS=ON Valgrind Massif\n snapshots are taken before and after each Time invocation. The\n resulting memory profile can be plotted using\n core/utils/plotMassifMemoryUsage.py\n\n \n This class must only be used to time functions that are\n   called only within the main program.  It may not be used\n   in pre-program setup or post-program teardown!");
		cl.def( pybind11::init( [](class Teuchos::Time & a0){ return new Teuchos::TimeMonitor(a0); } ), "doc" , pybind11::arg("timer"));
		cl.def( pybind11::init<class Teuchos::Time &, bool>(), pybind11::arg("timer"), pybind11::arg("reset") );

		cl.def( pybind11::init( [](Teuchos::TimeMonitor const &o){ return new Teuchos::TimeMonitor(o); } ) );
		cl.def_static("getNewTimer", (class Teuchos::RCP<class Teuchos::Time> (*)(const std::string &)) &Teuchos::TimeMonitor::getNewTimer, "Return a new timer with the given name (class method).\n\n Call getNewCounter() or this method if you want to create a new\n named timer, and you would like TimeMonitor to track the timer\n for later computation of global statistics over processes.\n\n This method wraps getNewCounter() (inherited from the base\n class) for backwards compatibiity.\n\nC++: Teuchos::TimeMonitor::getNewTimer(const std::string &) --> class Teuchos::RCP<class Teuchos::Time>", pybind11::arg("name"));
		cl.def_static("disableTimer", (void (*)(const std::string &)) &Teuchos::TimeMonitor::disableTimer, "Disable the timer with the given name.\n\n \"Disable\" means that the timer (Time instance) will ignore all\n calls to start(), stop(), and incrementNumCalls().  The effect\n will be as if the TimeMonitor had never touched the timer.\n\n If the timer with the given name does not exist (was never\n created using getNewCounter() or getNewTimer()), then this\n method throws std::invalid_argument.  Otherwise, it disables the\n timer.  This effect lasts until the timer is cleared or until\n the timer is enabled, either by calling enableTimer() (see\n below) or by calling the Time instance's enable() method.\n\n Disabling a timer does not exclude it from the list of\n timers printed by summarize() or report().\n\nC++: Teuchos::TimeMonitor::disableTimer(const std::string &) --> void", pybind11::arg("name"));
		cl.def_static("enableTimer", (void (*)(const std::string &)) &Teuchos::TimeMonitor::enableTimer, "Enable the timer with the given name.\n\n If the timer with the given name does not exist (was never\n created using getNewCounter() or getNewTimer()), then this\n method throws std::invalid_argument.  Otherwise, it undoes the\n effect of disableTimer() on the timer with the given name.  If\n the timer with the given name was not disabled, then this method\n does nothing.\n\nC++: Teuchos::TimeMonitor::enableTimer(const std::string &) --> void", pybind11::arg("name"));
		cl.def_static("zeroOutTimers", (void (*)()) &Teuchos::TimeMonitor::zeroOutTimers, "Reset all global timers to zero.\n\n This method only affects Time objects created by getNewCounter()\n or getNewTimer().\n\n \n None of the timers must currently be running.\n\nC++: Teuchos::TimeMonitor::zeroOutTimers() --> void");
		cl.def_static("getValidReportParameters", (class Teuchos::RCP<const class Teuchos::ParameterList> (*)()) &Teuchos::TimeMonitor::getValidReportParameters, "Default parameters (with validators) for report().\n\nC++: Teuchos::TimeMonitor::getValidReportParameters() --> class Teuchos::RCP<const class Teuchos::ParameterList>");
	}
	{ // Teuchos::SyncTimeMonitor file:Teuchos_TimeMonitor.hpp line:769
		pybind11::class_<Teuchos::SyncTimeMonitor, Teuchos::RCP<Teuchos::SyncTimeMonitor>, Teuchos::TimeMonitor> cl(M("Teuchos"), "SyncTimeMonitor", "A TimeMonitor that waits at a MPI barrier before destruction.");
		cl.def( pybind11::init( [](class Teuchos::Time & a0, class Teuchos::Ptr<const class Teuchos::Comm<int> > const & a1){ return new Teuchos::SyncTimeMonitor(a0, a1); } ), "doc" , pybind11::arg("timer"), pybind11::arg("comm"));
		cl.def( pybind11::init<class Teuchos::Time &, class Teuchos::Ptr<const class Teuchos::Comm<int> >, bool>(), pybind11::arg("timer"), pybind11::arg("comm"), pybind11::arg("reset") );

		cl.def( pybind11::init( [](Teuchos::SyncTimeMonitor const &o){ return new Teuchos::SyncTimeMonitor(o); } ) );
	}
	{ // Teuchos::TimeMonitorSurrogateImpl file:Teuchos_TimeMonitor.hpp line:812
		pybind11::class_<Teuchos::TimeMonitorSurrogateImpl, Teuchos::RCP<Teuchos::TimeMonitorSurrogateImpl>, PyCallBack_Teuchos_TimeMonitorSurrogateImpl> cl(M("Teuchos"), "TimeMonitorSurrogateImpl", "Implementation of TimeMonitorSurrogate that invokes TimeMonitor.\n \n\n Users should not use this class or rely on it in any way.\n   It is an implementation detail.\n\n Please refer to the documentation of\n TimeMonitorSurrogateImplInserter and TimeMonitorSurrogate for an\n explanation of the purpose of this class.");
		cl.def( pybind11::init( [](PyCallBack_Teuchos_TimeMonitorSurrogateImpl const &o){ return new PyCallBack_Teuchos_TimeMonitorSurrogateImpl(o); } ) );
		cl.def( pybind11::init( [](Teuchos::TimeMonitorSurrogateImpl const &o){ return new Teuchos::TimeMonitorSurrogateImpl(o); } ) );
		cl.def( pybind11::init( [](){ return new Teuchos::TimeMonitorSurrogateImpl(); }, [](){ return new PyCallBack_Teuchos_TimeMonitorSurrogateImpl(); } ) );
		cl.def("assign", (class Teuchos::TimeMonitorSurrogateImpl & (Teuchos::TimeMonitorSurrogateImpl::*)(const class Teuchos::TimeMonitorSurrogateImpl &)) &Teuchos::TimeMonitorSurrogateImpl::operator=, "C++: Teuchos::TimeMonitorSurrogateImpl::operator=(const class Teuchos::TimeMonitorSurrogateImpl &) --> class Teuchos::TimeMonitorSurrogateImpl &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::TimeMonitorSurrogateImplInserter file:Teuchos_TimeMonitor.hpp line:860
		pybind11::class_<Teuchos::TimeMonitorSurrogateImplInserter, Teuchos::RCP<Teuchos::TimeMonitorSurrogateImplInserter>> cl(M("Teuchos"), "TimeMonitorSurrogateImplInserter", "Injects run-time dependency of a class on TimeMonitor.\n \n\n Users should not use this class or rely on it in any way.\n   It is an implementation detail.\n\n \n\n Classes and functions with the name \"TimeMonitorSurrogate\" in them\n let CommandLineProcessor optionally call TimeMonitor::summarize(),\n without needing to know that the TimeMonitor class exists.  This\n allows Teuchos to put CommandLineProcessor in a separate package\n from TimeMonitor.  We want to do this because TimeMonitor depends\n on Comm, and is therefore in the TeuchosComm subpackage (which\n depends on TeuchosCore), but CommandLineProcessor is in a\n different subpackage which does not depend on Comm.\n\n The TimeMonitorSurrogateImplInserter class' constructor ensures\n that CommandLineProcessor gets informed about TimeMonitor even\n before the program starts executing main().  This happens\n automatically, without changes to main(), because we declare an\n instance of this class in the header file.  If the TeuchosComm\n subpackage was built and its libraries were linked in,\n CommandLineProcessor will know about TimeMonitor.\n\n \n\n This is an instance of the\n Dependency injection\n design pattern.  CommandLineProcessor is not supposed to know\n about TimeMonitor, because CommandLineProcessor's subpackage does\n not depend on TimeMonitor's subpackage.  Thus,\n CommandLineProcessor interacts with TimeMonitor through the\n TimeMonitorSurrogate interface.  TimeMonitorSurrogateImplInserter\n \"injects\" the dependency at run time, if the TeuchosComm\n subpackage was enabled and the application linked with its\n libraries.\n\n Teuchos developers could imitate the pattern of this class in\n order to use TimeMonitor's class methods (such as summarize())\n from any other class that does not depend on the TeuchosComm\n subpackage.");
		cl.def( pybind11::init( [](){ return new Teuchos::TimeMonitorSurrogateImplInserter(); } ) );
		cl.def( pybind11::init( [](Teuchos::TimeMonitorSurrogateImplInserter const &o){ return new Teuchos::TimeMonitorSurrogateImplInserter(o); } ) );
	}
	{ // Teuchos::WorkspaceStore file:Teuchos_Workspace.hpp line:267
		pybind11::class_<Teuchos::WorkspaceStore, Teuchos::RCP<Teuchos::WorkspaceStore>> cl(M("Teuchos"), "WorkspaceStore", "Workspace encapsulation class.\n\n Base class for objects that allocate a huge block of memory\n at once and then allow RawWorkspace (an hense Workspace<T>) objects to be created\n that make use of this memory in a stack-like fasion.  The classes WorkspaceStore\n and RawWorkspace work closely together and are useless on their own.\n\n Through this interface, a client can not initialize or resize the size of the\n available workspace and can not directly instantiate objects of this type.\n Instead it must create a derived WorkspaceStoreInitializeable object defined later.");
		cl.def("num_bytes_total", (unsigned long (Teuchos::WorkspaceStore::*)() const) &Teuchos::WorkspaceStore::num_bytes_total, "Return the total number of bytes that where initially allocated.\n\nC++: Teuchos::WorkspaceStore::num_bytes_total() const --> unsigned long");
		cl.def("num_bytes_remaining", (unsigned long (Teuchos::WorkspaceStore::*)() const) &Teuchos::WorkspaceStore::num_bytes_remaining, "Return the number of bytes remaining currently.\n\nC++: Teuchos::WorkspaceStore::num_bytes_remaining() const --> unsigned long");
		cl.def("num_static_allocations", (int (Teuchos::WorkspaceStore::*)() const) &Teuchos::WorkspaceStore::num_static_allocations, "Return the number of static memory allocations granted thus far.\n This is the number of memory allocations requested by the creation\n of RawWorkspace objects where there was sufficient preallocated memory\n to satisfy the request.\n\nC++: Teuchos::WorkspaceStore::num_static_allocations() const --> int");
		cl.def("num_dyn_allocations", (int (Teuchos::WorkspaceStore::*)() const) &Teuchos::WorkspaceStore::num_dyn_allocations, "Return the number of dynamic memory allocations granted thus far.\n This is the number of memory allocations requested by the creation\n of RawWorkspace objects where there was not sufficient preallocated memory\n to satisfy the request and dynamic memory had to be created.\n\nC++: Teuchos::WorkspaceStore::num_dyn_allocations() const --> int");
		cl.def("num_current_bytes_total", (unsigned long (Teuchos::WorkspaceStore::*)()) &Teuchos::WorkspaceStore::num_current_bytes_total, "Return the total number of bytes currently allocated..  This is the\n total number of bytes currently being used.\n\nC++: Teuchos::WorkspaceStore::num_current_bytes_total() --> unsigned long");
		cl.def("num_max_bytes_needed", (unsigned long (Teuchos::WorkspaceStore::*)() const) &Teuchos::WorkspaceStore::num_max_bytes_needed, "Return the maximum storage in bytes needed.  This is the maximum\n total amount of * storage that was needed at any one time.\n\nC++: Teuchos::WorkspaceStore::num_max_bytes_needed() const --> unsigned long");
	}
	{ // Teuchos::WorkspaceStoreInitializeable file:Teuchos_Workspace.hpp line:330
		pybind11::class_<Teuchos::WorkspaceStoreInitializeable, Teuchos::RCP<Teuchos::WorkspaceStoreInitializeable>, Teuchos::WorkspaceStore> cl(M("Teuchos"), "WorkspaceStoreInitializeable", "WorkspaceStore class that can be used to actually reinitialize memory.\n\n The client can create concrete instances of this type and initialize\n the memory used.  The client should call initialize(num_bytes) to set the number\n of bytes to allocate where num_bytes should be large enough to satisfy all but\n the largests of memory request needs.");
		cl.def( pybind11::init( [](){ return new Teuchos::WorkspaceStoreInitializeable(); } ), "doc" );
		cl.def( pybind11::init<unsigned long>(), pybind11::arg("num_bytes") );

		cl.def("initialize", (void (Teuchos::WorkspaceStoreInitializeable::*)(unsigned long)) &Teuchos::WorkspaceStoreInitializeable::initialize, "Set the size block of memory to be given as workspace.\n\n If there are any instantiated RawWorkspace objects then this\n function willl throw an std::exception.  It must be called before\n any RawWorkspace objects are created.\n\nC++: Teuchos::WorkspaceStoreInitializeable::initialize(unsigned long) --> void", pybind11::arg("num_bytes"));
	}
}
