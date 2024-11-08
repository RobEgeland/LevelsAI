#include <iostream>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <thread>
#include <atomic>

std::atomic<bool> isRunning(true);

void WaitForEnter() {
    std::cin.get();  // Waits until Enter is pressed
    isRunning = false;  // Set flag to stop the main loop
}

void InitializeWASAPI() {
    HRESULT hr = CoInitialize(NULL);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM library." << std::endl;
        return;
    }

    IMMDeviceEnumerator* deviceEnumerator = NULL;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&deviceEnumerator));
    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator." << std::endl;
        return;
    }

    IMMDevice* defaultDevice = NULL;
    hr = deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &defaultDevice);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default audio endpoint." << std::endl;
        deviceEnumerator->Release();
        return;
    }

    IAudioClient* audioClient = NULL;
    hr = defaultDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&audioClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to activate audio client." << std::endl;
        defaultDevice->Release();
        deviceEnumerator->Release();
        return;
    }

    WAVEFORMATEX* waveFormat;
    hr = audioClient->GetMixFormat(&waveFormat);
    if (FAILED(hr)) {
        std::cerr << "Failed to get mix format." << std::endl;
        audioClient->Release();
        defaultDevice->Release();
        deviceEnumerator->Release();
        return;
    }

    hr = audioClient->Initialize(
    AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_LOOPBACK,  // Enable loopback mode
    10000000,                      // Buffer duration (1 second)
    0,
    waveFormat,
    NULL
    );


    //hr = audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, waveFormat, NULL);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client with custom format. HRESULT: " << std::hex << hr << std::endl;
        audioClient->Release();
        defaultDevice->Release();
        deviceEnumerator->Release();
        return;
    }
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client." << std::endl;
        CoTaskMemFree(waveFormat);
        audioClient->Release();
        defaultDevice->Release();
        deviceEnumerator->Release();
        return;
    }

    IAudioCaptureClient* captureClient = NULL;
    hr = audioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&captureClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to get audio capture client." << std::endl;
        CoTaskMemFree(waveFormat);
        audioClient->Release();
        defaultDevice->Release();
        deviceEnumerator->Release();
        return;
    }

    hr = audioClient->Start();
    if (FAILED(hr)) {
        std::cerr << "Failed to start audio client." << std::endl;
        captureClient->Release();
        CoTaskMemFree(waveFormat);
        audioClient->Release();
        defaultDevice->Release();
        deviceEnumerator->Release();
        return;
    }

    std::cout << "Monitoring audio data. Press Enter to stop..." << std::endl;

    std::thread waitForEnterThread(WaitForEnter);

    while (isRunning) {
        UINT32 packetLength = 0;
        hr = captureClient->GetNextPacketSize(&packetLength);

        if (packetLength > 0) {
            BYTE* data;
            UINT32 numFramesAvailable;
            DWORD flags;
            hr = captureClient->GetBuffer(&data, &numFramesAvailable, &flags, NULL, NULL);
            if (SUCCEEDED(hr)) {
                if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                std::cout << "Silent packet detected" << std::endl;
                }
                else {
                    std::cout << "Captured audio data, frames: " << numFramesAvailable << std::endl;

                }
                captureClient->ReleaseBuffer(numFramesAvailable);
                int16_t* samples = reinterpret_cast<int16_t*>(data);  // Assuming 16-bit PCM data
                int numSamples = numFramesAvailable * waveFormat->nChannels;

                int64_t sum = 0;
                for (int i = 0; i < numSamples; ++i) {
                    sum += abs(samples[i]);
                }
                double averageAmplitude = static_cast<double>(sum) / numSamples;
                std::cout << "Average Amplitude: " << averageAmplitude << std::endl;
            }
        }
    }


    audioClient->Stop();
    captureClient->Release();
    CoTaskMemFree(waveFormat);
    audioClient->Release();
    defaultDevice->Release();
    deviceEnumerator->Release();
    CoUninitialize();

    // Wait for the input thread to finish before exiting
    waitForEnterThread.join();

}
int main() {
    InitializeWASAPI();
    //std::cout << "Hello, World!" << std::endl;
    return 0;
}