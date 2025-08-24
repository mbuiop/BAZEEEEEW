#include "GameHUD.h"
#include "Blueprint/UserWidget.h"
#include "Engine/Canvas.h"
#include "BBGameMode.h"
#include "FighterChar.h"
#include "Components/HealthComp.h"

AGameHUD::AGameHUD()
{
    
}

void AGameHUD::BeginPlay()
{
    Super::BeginPlay();
    
    if (PlayerHUDClass)
    {
        PlayerHUD = CreateWidget<UUserWidget>(GetWorld(), PlayerHUDClass);
        if (PlayerHUD)
        {
            PlayerHUD->AddToViewport();
        }
    }
}

void AGameHUD::DrawHUD()
{
    Super::DrawHUD();
    
    DrawHealthBar();
    DrawScore();
    DrawWaveInfo();
}

void AGameHUD::DrawHealthBar()
{
    APlayerController* PC = GetOwningPlayerController();
    if (!PC) return;
    
    AFighterChar* Fighter = Cast<AFighterChar>(PC->GetPawn());
    if (!Fighter || !Fighter->HealthComponent) return;
    
    float HealthPercent = Fighter->HealthComponent->CurrentHealth / Fighter->HealthComponent->MaxHealth;
    FVector2D ScreenDimensions = FVector2D(Canvas->SizeX, Canvas->SizeY);
    
    FVector2D HealthBarPosition(50, ScreenDimensions.Y - 50);
    FVector2D HealthBarSize(200, 20);
    
    FCanvasTileItem HealthBarBackground(HealthBarPosition, FVector2D(HealthBarSize.X, HealthBarSize.Y), FLinearColor::Gray);
    Canvas->DrawItem(HealthBarBackground);
    
    FCanvasTileItem HealthBarForeground(HealthBarPosition, FVector2D(HealthBarSize.X * HealthPercent, HealthBarSize.Y), FLinearColor::Green);
    Canvas->DrawItem(HealthBarForeground);
}

void AGameHUD::DrawScore()
{
    APlayerController* PC = GetOwningPlayerController();
    if (!PC) return;
    
    ABBGameMode* GameMode = Cast<ABBGameMode>(GetWorld()->GetAuthGameMode());
    if (!GameMode) return;
    
    FVector2D ScreenDimensions = FVector2D(Canvas->SizeX, Canvas->SizeY);
    FVector2D ScorePosition(ScreenDimensions.X - 200, 50);
    
    FString ScoreString = FString::Printf(TEXT("Score: %d"), GameMode->BlocksDestroyed);
    DrawText(ScoreString, FLinearColor::White, ScorePosition.X, ScorePosition.Y);
}

void AGameHUD::DrawWaveInfo()
{
    APlayerController* PC = GetOwningPlayerController();
    if (!PC) return;
    
    ABBGameMode* GameMode = Cast<ABBGameMode>(GetWorld()->GetAuthGameMode());
    if (!GameMode) return;
    
    FVector2D ScreenDimensions = FVector2D(Canvas->SizeX, Canvas->SizeY);
    FVector2D WavePosition(50, 50);
    
    FString WaveString = FString::Printf(TEXT("Wave: %d"), GameMode->CurrentWave);
    DrawText(WaveString, FLinearColor::Yellow, WavePosition.X, WavePosition.Y);
}
