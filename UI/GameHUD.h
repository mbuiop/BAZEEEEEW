#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "GameHUD.generated.h"

UCLASS()
class BLOCKBREAKER3D_API AGameHUD : public AHUD
{
    GENERATED_BODY()
    
public:
    AGameHUD();
    
    virtual void DrawHUD() override;
    virtual void BeginPlay() override;
    
    UPROPERTY(EditDefaultsOnly, Category = "UI")
    TSubclassOf<class UUserWidget> PlayerHUDClass;
    
    UPROPERTY()
    class UUserWidget* PlayerHUD;

protected:
    void DrawHealthBar();
    void DrawScore();
    void DrawWaveInfo();
};
