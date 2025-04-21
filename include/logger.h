#pragma once
#include <fstream>
#include <mutex>
#include <string>
#include <chrono>
#include <ctime>
#include <sstream>

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
public:
    // Singleton access
    static Logger& instance() {
        static Logger inst("app.log");
        return inst;
    }

    // Log something
    void log(LogLevel level, const std::string& msg,
             const char* file, int line) {
        std::lock_guard<std::mutex> lock(mtx_);
        stream_ << timestamp()
                << " [" << toString(level) << "] "
                << file << ":" << line << " "
                << msg << "\n";
        stream_.flush();
    }

    // Print a line of asterisks
    void printAsteriskLine() {
        std::lock_guard<std::mutex> lock(mtx_);
        stream_ << std::string(30, '*') << "\n";
        stream_.flush();
    }

    // Set the minimum level to output
    void setLevel(LogLevel level) { level_ = level; }

private:
    std::ofstream stream_;
    std::mutex mtx_;
    LogLevel level_ = LogLevel::DEBUG;

    Logger(const std::string& filename)
        : stream_(filename, std::ios::app) {}

    // get a formatted timestamp
    std::string timestamp() {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto t = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        std::tm bt;
        #ifdef _WIN32
            localtime_s(&bt, &t);
        #else
            localtime_r(&t, &bt);
        #endif

        std::ostringstream oss;
        oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    // convert enum to text
    const char* toString(LogLevel lvl) {
        switch (lvl) {
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::INFO:    return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR:   return "ERROR";
        }
        return "UNK";
    }
};

// Macros for convenience
#define LOG_DEBUG(msg)   Logger::instance().log(LogLevel::DEBUG,   msg, __FILE__, __LINE__)
#define LOG_INFO(msg)    Logger::instance().log(LogLevel::INFO,    msg, __FILE__, __LINE__)
#define LOG_WARNING(msg) Logger::instance().log(LogLevel::WARNING, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg)   Logger::instance().log(LogLevel::ERROR,   msg, __FILE__, __LINE__)
