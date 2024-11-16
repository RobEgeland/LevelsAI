#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    try {
        // Initialize env
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        std::cout << "ONNX Runtime initialized successfully!" << std::endl;
        
        // Print version info
        std::cout << "ONNX Runtime version: " << ORT_API_VERSION_STRING << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}