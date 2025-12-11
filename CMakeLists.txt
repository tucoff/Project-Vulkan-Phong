cmake_minimum_required(VERSION 3.20)
project(VulkanGame 
    VERSION 1.0.0
    DESCRIPTION "A Vulkan-based Game Engine"
    LANGUAGES CXX
)

# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Project structure
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Project paths
set(SRC_DIR ${PROJECT_ROOT_DIR}/src)
set(SHADER_DIR ${PROJECT_ROOT_DIR}/shaders)
set(TEXTURE_DIR ${PROJECT_ROOT_DIR}/textures)

# Dependencies
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# Collect source files
file(GLOB_RECURSE PROJECT_SOURCES
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.h"
)

# Create main executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${SRC_DIR}
    ${Vulkan_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
)

# Platform-specific configurations
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        VK_USE_PLATFORM_WIN32_KHR
    )
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        VK_USE_PLATFORM_XCB_KHR
    )
endif()

# Shader compilation
find_program(GLSLC_EXECUTABLE glslc HINTS $ENV{VULKAN_SDK}/bin)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc shader compiler not found!")
endif()

# Function to compile shaders
function(compile_shader SHADER_SOURCE)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_SOURCE} LAST_EXT)
    # Remove the dot from extension
    string(SUBSTRING ${SHADER_EXT} 1 -1 SHADER_EXT_CLEAN)
    set(SPIRV_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${SHADER_NAME}_${SHADER_EXT_CLEAN}.spv")
    
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders"
        COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE} -o ${SPIRV_OUTPUT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling shader ${SHADER_NAME}_${SHADER_EXT_CLEAN}"
        VERBATIM
    )
    list(APPEND SHADER_BINARIES ${SPIRV_OUTPUT})
    set(SHADER_BINARIES ${SHADER_BINARIES} PARENT_SCOPE)
endfunction()

# Compile all shaders
file(GLOB SHADER_SOURCES
    "${SHADER_DIR}/*.vert"
    "${SHADER_DIR}/*.frag"
    "${SHADER_DIR}/*.comp"
)

set(SHADER_BINARIES "")
foreach(SHADER ${SHADER_SOURCES})
    compile_shader(${SHADER})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_BINARIES})
add_dependencies(${PROJECT_NAME} shaders)

# Smart asset sync that only copies when needed
add_custom_target(sync_assets ALL
    COMMENT "Checking and syncing assets to build directory"
    
    # Ensure output directories exist
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/textures
    
    # Sync models (copy if different or missing)
    COMMAND ${CMAKE_COMMAND} -E echo "Syncing models..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_ROOT_DIR}/models ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models
    
    # Sync textures (copy if different or missing)
    COMMAND ${CMAKE_COMMAND} -E echo "Syncing textures..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_ROOT_DIR}/textures ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/textures
    
    VERBATIM
)

# Add individual file dependencies so CMake knows when to re-sync
file(GLOB_RECURSE MODEL_FILES "${PROJECT_ROOT_DIR}/models/*")
file(GLOB_RECURSE TEXTURE_FILES "${PROJECT_ROOT_DIR}/textures/*")

# Create custom commands for each file type
foreach(MODEL_FILE ${MODEL_FILES})
    get_filename_component(MODEL_NAME ${MODEL_FILE} NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/${MODEL_NAME}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_FILE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/${MODEL_NAME}
        DEPENDS ${MODEL_FILE}
        COMMENT "Updating model: ${MODEL_NAME}"
    )
endforeach()

foreach(TEXTURE_FILE ${TEXTURE_FILES})
    get_filename_component(TEXTURE_NAME ${TEXTURE_FILE} NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/textures/${TEXTURE_NAME}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEXTURE_FILE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/textures/${TEXTURE_NAME}
        DEPENDS ${TEXTURE_FILE}
        COMMENT "Updating texture: ${TEXTURE_NAME}"
    )
endforeach()

add_dependencies(${PROJECT_NAME} sync_assets)

# Development options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _DEBUG
        VK_ENABLE_VALIDATION_LAYERS
    )
    
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /W4)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/
    DESTINATION bin/shaders
)

install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/textures/
    DESTINATION bin/textures
)