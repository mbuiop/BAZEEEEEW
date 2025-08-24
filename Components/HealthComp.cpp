#include "HealthComp.h"

UHealthComp::UHealthComp()
{
    PrimaryComponentTick.bCanEverTick = true;
    MaxHealth = 100.0f;
    CurrentHealth = MaxHealth;
}

void UHealthComp::BeginPlay()
{
    Super::BeginPlay();
}

void UHealthComp::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void UHealthComp::TakeDamage(float DamageAmount)
{
    CurrentHealth = FMath::Clamp(CurrentHealth - DamageAmount, 0.0f, MaxHealth);
    
    OnHealthChanged.Broadcast(CurrentHealth / MaxHealth);
    
    if (CurrentHealth <= 0.0f)
    {
        OnDeath.Broadcast();
    }
}

void UHealthComp::Heal(float HealAmount)
{
    CurrentHealth = FMath::Clamp(CurrentHealth + HealAmount, 0.0f, MaxHealth);
    OnHealthChanged.Broadcast(CurrentHealth / MaxHealth);
}
