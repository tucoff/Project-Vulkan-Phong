// Region: Includes
// This section includes necessary headers and defines platform-specific macros for Vulkan and GLFW.
// It also sets up the application window and prepares for Vulkan initialization.
// The code is structured to be compatible with both Windows and Linux platforms, using conditional compilation.
#pragma region Includes

// utils.cpp
#include "utils.h"          // Include the header file for this module

#define GLFW_INCLUDE_VULKAN         // Define this to include Vulkan-specific headers with GLFW
#include <GLFW/glfw3.h>             // Include GL framework for window management and input handling
#include <iostream>                 // For standard input/output operations (e.g., std::cerr)
#include <fstream>                  // For file stream operations (not used in this code, but often included for file I/O)
#include <stdexcept>                // For standard exception handling (e.g., std::runtime_error)
#include <vector>                   // For using std::vector dynamic arrays
#include <cstring>                  // For C-style string manipulation (e.g., strcmp)
#include <cstdlib>                  // For general utilities (e.g., EXIT_SUCCESS, EXIT_FAILURE)
#include <optional>                 // For using std::optional to represent potentially absent values
#include <set>                      // For using std::set to store unique values
#include <cstdint>                  //Necessary for uint32_t
#include <limits>                   //Necessary for std::numeric_limits
#include <algorithm>                //Necessary for std::clamp

#pragma endregion

// Region: Proxy Functions
// This section defines proxy functions for Vulkan debug utilities.
// These functions are used to create and destroy debug messengers, which are essential for debugging Vulkan
// applications. They are defined as proxies because they call the actual Vulkan functions dynamically at runtime.
# pragma region Proxy Functions

// Proxy function to create a debug utility messenger.
// vkCreateDebugUtilsMessengerEXT is an extension function, so its address must be loaded at runtime.
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    // Get the address of the vkCreateDebugUtilsMessengerEXT function from the Vulkan instance.
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    // If the function pointer is valid, call the function.
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    // Otherwise, return an error indicating the extension is not present.
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Proxy function to destroy a debug utility messenger.
// vkDestroyDebugUtilsMessengerEXT is an extension function, so its address must be loaded at runtime.
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    // Get the address of the vkDestroyDebugUtilsMessengerEXT function from the Vulkan instance.
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    // If the function pointer is valid, call the function.
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

# pragma endregion