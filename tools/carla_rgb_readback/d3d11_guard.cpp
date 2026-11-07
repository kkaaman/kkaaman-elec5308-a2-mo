// Local CARLA diagnostic shim: serialize access to the D3D11 immediate context.
// The real D3D11 implementation is loaded by its absolute System32 path.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <d3d11_4.h>
#include <cstdio>
#include <cwchar>

static HMODULE self_module = nullptr;
static HMODULE system_module = nullptr;
static INIT_ONCE load_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK load_system_d3d11(PINIT_ONCE, PVOID, PVOID*) {
    wchar_t path[MAX_PATH] = {};
    UINT count = GetSystemDirectoryW(path, MAX_PATH);
    if (!count || count + 12 >= MAX_PATH) return FALSE;
    std::wcscat(path, L"\\d3d11.dll");
    system_module = LoadLibraryW(path);
    return system_module != nullptr;
}

static void protect_context(ID3D11Device* device, UINT original_flags, UINT effective_flags) {
    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);
    ID3D11Multithread* multithread = nullptr;
    const GUID iid = {0x9b7e4e00, 0x342c, 0x4106,
                      {0xa1, 0x9f, 0x4f, 0x27, 0x04, 0xf6, 0x89, 0xf0}};
    HRESULT query = context ? context->QueryInterface(iid,
                             reinterpret_cast<void**>(&multithread)) : E_POINTER;
    BOOL before = FALSE;
    BOOL after = FALSE;
    if (SUCCEEDED(query) && multithread) {
        before = multithread->GetMultithreadProtected();
        multithread->SetMultithreadProtected(TRUE);
        after = multithread->GetMultithreadProtected();
        multithread->Release();
    }
    if (context) context->Release();
    wchar_t path[MAX_PATH] = {};
    if (GetModuleFileNameW(self_module, path, MAX_PATH)) {
        wchar_t* slash = std::wcsrchr(path, L'\\');
        if (slash) {
            *(slash + 1) = L'\0';
            std::wcscat(path, L"carla-d3d11-guard.jsonl");
            HANDLE file = CreateFileW(path, FILE_APPEND_DATA,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL, nullptr);
            if (file != INVALID_HANDLE_VALUE) {
                char line[256] = {};
                int size = std::snprintf(line, sizeof(line),
                    "{\"flags_before\":%u,\"flags_after\":%u,\"query_hresult\":%ld,"
                    "\"protected_before\":%s,\"protected_after\":%s}\n",
                    original_flags, effective_flags, static_cast<long>(query),
                    before ? "true" : "false", after ? "true" : "false");
                DWORD written = 0;
                if (size > 0) WriteFile(file, line, static_cast<DWORD>(size), &written, nullptr);
                CloseHandle(file);
            }
        }
    }
}

extern "C" HRESULT WINAPI GuardedD3D11CreateDevice(
    IDXGIAdapter* adapter, D3D_DRIVER_TYPE type, HMODULE software, UINT flags,
    const D3D_FEATURE_LEVEL* levels, UINT level_count, UINT sdk,
    ID3D11Device** device, D3D_FEATURE_LEVEL* selected,
    ID3D11DeviceContext** context) {
    if (!InitOnceExecuteOnce(&load_once, load_system_d3d11, nullptr, nullptr)) return E_FAIL;
    using Function = HRESULT (WINAPI*)(IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT,
        const D3D_FEATURE_LEVEL*, UINT, UINT, ID3D11Device**, D3D_FEATURE_LEVEL*,
        ID3D11DeviceContext**);
    auto function = reinterpret_cast<Function>(GetProcAddress(system_module, "D3D11CreateDevice"));
    if (!function) return E_FAIL;
    ID3D11Device* local_device = nullptr;
    ID3D11Device** actual_device = device ? device : &local_device;
    UINT effective_flags = flags & ~D3D11_CREATE_DEVICE_SINGLETHREADED;
    HRESULT result = function(adapter, type, software, effective_flags, levels,
                              level_count, sdk, actual_device, selected, context);
    if (SUCCEEDED(result) && *actual_device) {
        protect_context(*actual_device, flags, effective_flags);
    }
    if (local_device) local_device->Release();
    return result;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        self_module = instance;
        DisableThreadLibraryCalls(instance);
    }
    return TRUE;
}
