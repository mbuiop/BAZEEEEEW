#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "HealthComp.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnDeathDelegate);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnHealthChangedDelegate, float, HealthPercent);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class BLOCKBREAKER3D_API UHealthComp : public UActorComponent
{
    GENERATED_BODY()

public:
    UHealthComp();
    
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
    
    UFUNCTION(BlueprintCallable)
    void TakeDamage(float DamageAmount);
    
    UFUNCTION(BlueprintCallable)
    void Heal(float HealAmount);
    
    UPROPERTY(BlueprintAssignable)
    FOnDeathDelegate OnDeath;
    
    UPROPERTY(BlueprintAssignable)
    FOnHealthChangedDelegate OnHealthChanged;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Health")
    float MaxHealth;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Health")
    float CurrentHealth;

protected:
    virtual void BeginPlay() override;
};
