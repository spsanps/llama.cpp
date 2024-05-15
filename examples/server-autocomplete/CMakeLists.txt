set(TARGET autocomplete)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET} autocomplete.cpp token_cache.h json.hpp httplib.h rolling.h search_info.cpp search_info.h task_management.h)
install(TARGETS ${TARGET} RUNTIME)
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_11)
