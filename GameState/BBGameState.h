#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameStateBase.h"
#include "BBGameState.generated.h"

UCLASS()
class BLOCKBREAKER3D_API ABBGameState : public AGameStateBase
{
    GENERATED_BODY()
    
public:
    ABBGameState();
    
    UPROPERTY(ReplicatedUsing=OnRep_PlayerScore, BlueprintReadOnly, Category = "Game State")
    int32 PlayerScore;
    
    UPROPERTY(ReplicatedUsing=OnRep_GameTime, BlueprintReadOnly, Category = "Game State")
    float GameTime;
    
    UFUNCTION(BlueprintCallable)
    void AddScore(int32 ScoreToAdd);
    
    virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override;

protected:
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaTime) override;
    
    UFUNCTION()
    void OnRep_PlayerScore();
    
    UFUNCTION()
    void OnRep_GameTime();
};
