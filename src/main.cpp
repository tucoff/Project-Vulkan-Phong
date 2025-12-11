#include "game.h"
#include "compute.h"

#include <iostream>
#include <string>

int main() 
{
    std::cout << "=== VULKAN GAME PROJECT ===" << std::endl;
    std::cout << "Controles:" << std::endl;
    std::cout << "- ESC: Sair do programa" << std::endl;
    std::cout << "- Tecla 1: Mudar para Game (3D)" << std::endl;
    std::cout << "- Tecla 2: Mudar para Compute (Partículas)" << std::endl;
    std::cout << "===========================" << std::endl << std::endl;
    
    std::cout << "Iniciando com Compute (Partículas)..." << std::endl;
    compute();
    return 0;
}