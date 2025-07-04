#include <ScreenCapture.au3>

; Create ss folder if it doesn't exist
If Not FileExists(@ScriptDir & "\ss") Then
    DirCreate(@ScriptDir & "\ss")
EndIf

; Initialize counter
Global $iCount = 1

; Set hotkey for F6
HotKeySet("{F6}", "TakeScreenshot")

; Keep script running
While True
    Sleep(100)
WEnd

; Function to take screenshot
Func TakeScreenshot()
    Local $hBmp, $sFile

    ; Get the desktop dimensions
    Local $aSize = WinGetClientSize("[CLASS:Progman]")
    If @error Then Return

    ; Capture entire screen
    $hBmp = _ScreenCapture_Capture("", 0, 0, $aSize[0], $aSize[1])
    If @error Then Return

    ; Build file path
    $sFile = @ScriptDir & "\ss\screenshot_" & @YEAR & @MON & @MDAY & "_" & @HOUR & @MIN & @SEC & "_" & $iCount & ".png"

    ; Save screenshot
    _ScreenCapture_SaveImage($sFile, $hBmp)
    $iCount += 1
    Send("{PGDN}")
    ; Notify
    ;TrayTip("Screenshot Taken", "Saved to: " & $sFile, 2)
	
EndFunc
