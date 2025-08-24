#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "BBGameMode.generated.h"

UCLASS()
class BLOCKBREAKER3D_API ABBGameMode : public AGameModeBase
{
    GENERATED_BODY()
    
public:
    ABBGameMode();
    
    virtual void StartPlay() override;
    virtual void Tick(float DeltaTime) override;
    
    UFUNCTION(BlueprintCallable)
    void SpawnBlockWave();
    
    UFUNCTION(BlueprintCallable)
    void IncreaseDifficulty();
    
    UPROPERTY(EditDefaultsOnly, Category = "Block")
    TSubclassOf<class ABlockActor> BlockClass;
    
    UPROPERTY(EditDefaultsOnly, Category = "Projectile")
    TSubclassOf<class AProjectile> ProjectileClass;
    
    UPROPERTY(EditDefaultsOnly, Category = "Coin")
    TSubclassOf<class ACoin> CoinClass;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Game State")
    int32 CurrentWave;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Game State")
    int32 BlocksDestroyed;
    
    UPROPERTY(EditDefaultsOnly, Category = "Spawning")
    float SpawnInterval;
    
    UPROPERTY(EditDefaultsOnly, Category = "Spawning")
    float DifficultyInterval;

protected:
    FTimerHandle SpawnTimerHandle;
    FTimerHandle DifficultyTimerHandle;
    
    virtual void BeginPlay() override;
    void SetupTimers();
};
